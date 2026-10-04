#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from scapy.all import IP, UDP, Raw, rdpcap, send
    from scapy.packet import Packet
except ImportError as exc:
    raise SystemExit(
        "Scapy is required for UDP pcap replay. Install/use the repo environment with:\n"
        "  pixi run udp_pcap_replay --help\n"
        "or install it manually in the active Python environment."
    ) from exc


@dataclass(frozen=True)
class ReplayOptions:
    pcap_path: Path
    dst_ip: str
    dst_port: int | None
    src_ip: str | None
    src_port: int | None
    preserve_timing: bool
    fixed_rate_hz: float
    speed: float
    max_delay_s: float
    repeat: int
    loop_delay_s: float
    limit: int | None
    dedupe_window_s: float
    progress_interval_s: float
    verbose_packets: bool
    dry_run: bool
    quiet: bool


@dataclass(frozen=True)
class UdpReplayPacket:
    packet: Packet
    capture_time_s: float
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    payload_len: int


@dataclass(frozen=True)
class UdpReplayPacketBatch:
    packets: tuple[UdpReplayPacket, ...]


@dataclass(frozen=True)
class ReplaySummary:
    read_packets: int
    udp_packets: int
    selected_packets: int
    sent_packets: int


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed >= 65536:
        raise argparse.ArgumentTypeError("port must be in range 1..65535")
    return parsed


def _optional_limit(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("limit must be greater than 0")
    return parsed


def _repeat_count(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("repeat must be greater than or equal to 0")
    return parsed


def _packet_capture_time_s(packet: Packet) -> float:
    return float(packet.time)


def _payload_len(packet: Packet) -> int:
    if Raw in packet:
        return len(bytes(packet[Raw].load))
    return len(bytes(packet[UDP].payload))


def _normalize_udp_packet(packet: Packet, options: ReplayOptions) -> UdpReplayPacket | None:
    if IP not in packet or UDP not in packet:
        return None

    ip_packet = packet[IP].copy()
    udp_layer = ip_packet[UDP]

    if options.dst_ip:
        ip_packet.dst = options.dst_ip
    if options.src_ip is not None:
        ip_packet.src = options.src_ip
    if options.dst_port is not None:
        udp_layer.dport = options.dst_port
    if options.src_port is not None:
        udp_layer.sport = options.src_port

    del ip_packet.chksum
    del udp_layer.chksum
    del ip_packet.len
    del udp_layer.len

    return UdpReplayPacket(
        packet=ip_packet,
        capture_time_s=_packet_capture_time_s(packet),
        src_ip=str(ip_packet.src),
        src_port=int(udp_layer.sport),
        dst_ip=str(ip_packet.dst),
        dst_port=int(udp_layer.dport),
        payload_len=_payload_len(ip_packet),
    )


def _dedupe_key(packet: UdpReplayPacket) -> tuple[str, int, str, int, bytes]:
    payload = bytes(packet.packet[UDP].payload)
    return (packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port, payload)


def _select_udp_packets(raw_packets: Iterable[Packet], options: ReplayOptions) -> tuple[int, int, list[UdpReplayPacket]]:
    read_packets = 0
    udp_packets = 0
    selected_packets: list[UdpReplayPacket] = []
    recent_seen_s: dict[tuple[str, int, str, int, bytes], float] = {}

    for raw_packet in raw_packets:
        read_packets += 1

        # Some loopback captures are stored with a synthetic Ethernet header.
        # We replay at the IP layer, so Ether is intentionally discarded.
        # Keep the capture timestamp on the outer packet. Normalization copies
        # the IP layer and discards Ethernet without losing timestamp metadata.
        replay_packet = _normalize_udp_packet(raw_packet, options)
        if replay_packet is None:
            continue

        udp_packets += 1
        if options.dedupe_window_s > 0:
            key = _dedupe_key(replay_packet)
            previous_seen_s = recent_seen_s.get(key)
            if previous_seen_s is not None and replay_packet.capture_time_s - previous_seen_s <= options.dedupe_window_s:
                continue
            recent_seen_s[key] = replay_packet.capture_time_s

        selected_packets.append(replay_packet)
        if options.limit is not None and len(selected_packets) >= options.limit:
            break

    return read_packets, udp_packets, selected_packets


def _fixed_rate_packet_batches(
    selected_packets: list[UdpReplayPacket],
    *,
    fixed_rate_hz: float,
) -> list[UdpReplayPacketBatch]:
    if not selected_packets:
        return []
    if fixed_rate_hz <= 0.0:
        return [UdpReplayPacketBatch(packets=(packet,)) for packet in selected_packets]

    tick_s = 1.0 / fixed_rate_hz
    first_capture_time_s = selected_packets[0].capture_time_s
    batches: list[UdpReplayPacketBatch] = []
    current_bucket_index = 0
    current_packets: list[UdpReplayPacket] = []

    for replay_packet in selected_packets:
        offset_s = max(0.0, replay_packet.capture_time_s - first_capture_time_s)
        bucket_index = int(offset_s / tick_s)
        if not current_packets:
            current_bucket_index = bucket_index
            current_packets.append(replay_packet)
            continue
        if bucket_index == current_bucket_index:
            current_packets.append(replay_packet)
            continue
        batches.append(UdpReplayPacketBatch(packets=tuple(current_packets)))
        current_bucket_index = bucket_index
        current_packets = [replay_packet]

    if current_packets:
        batches.append(UdpReplayPacketBatch(packets=tuple(current_packets)))
    return batches


def _sleep_for_capture_delta(previous_time_s: float, current_time_s: float, options: ReplayOptions) -> None:
    if not options.preserve_timing:
        return

    delay_s = max(0.0, (current_time_s - previous_time_s) / options.speed)
    if options.max_delay_s > 0:
        delay_s = min(delay_s, options.max_delay_s)
    if delay_s > 0:
        time.sleep(delay_s)


def _sleep_for_fixed_rate_tick(batch_index: int, options: ReplayOptions) -> None:
    if options.fixed_rate_hz <= 0.0 or batch_index <= 1:
        return
    delay_s = (1.0 / options.fixed_rate_hz) / options.speed
    if delay_s > 0.0:
        time.sleep(delay_s)


def _format_progress_line(
    *,
    loop_index: int,
    loop_label: str,
    packet_index: int,
    packet_count: int,
    sent_packets: int,
    started_at_s: float,
    dry_run: bool,
) -> str:
    elapsed_s = max(1e-6, time.monotonic() - started_at_s)
    rate = sent_packets / elapsed_s
    percent = (packet_index / max(1, packet_count)) * 100.0
    action = "dry-run" if dry_run else "sent"
    return (
        f"\r[{loop_index}/{loop_label}] "
        f"{packet_index}/{packet_count} ({percent:5.1f}%) "
        f"{action}={sent_packets} rate={rate:7.1f} pkt/s"
    )


def _print_progress(line: str, *, final: bool = False) -> None:
    print(line, end="\n" if final else "", flush=True)


def replay_udp_pcap(options: ReplayOptions) -> ReplaySummary:
    if not options.pcap_path.is_file():
        raise FileNotFoundError(f"pcap file was not found: {options.pcap_path}")

    raw_packets = rdpcap(os.fspath(options.pcap_path))
    read_packets, udp_packets, selected_packets = _select_udp_packets(raw_packets, options)
    if not selected_packets:
        return ReplaySummary(
            read_packets=read_packets,
            udp_packets=udp_packets,
            selected_packets=0,
            sent_packets=0,
        )

    sent_packets = 0
    loop_index = 0
    fixed_rate_batches = _fixed_rate_packet_batches(selected_packets, fixed_rate_hz=options.fixed_rate_hz)
    started_at_s = time.monotonic()
    last_progress_s = 0.0
    try:
        while options.repeat == 0 or loop_index < options.repeat:
            loop_index += 1
            loop_label = "forever" if options.repeat == 0 else str(options.repeat)

            if options.fixed_rate_hz > 0.0:
                packet_index = 0
                for batch_index, replay_batch in enumerate(fixed_rate_batches, start=1):
                    _sleep_for_fixed_rate_tick(batch_index, options)
                    for replay_packet in replay_batch.packets:
                        packet_index += 1
                        sent_packets = _send_replay_packet(
                            options=options,
                            replay_packet=replay_packet,
                            loop_index=loop_index,
                            loop_label=loop_label,
                            sent_packets=sent_packets,
                        )
                        last_progress_s = _maybe_print_progress(
                            options=options,
                            loop_index=loop_index,
                            loop_label=loop_label,
                            packet_index=packet_index,
                            packet_count=len(selected_packets),
                            sent_packets=sent_packets,
                            started_at_s=started_at_s,
                            last_progress_s=last_progress_s,
                        )
            else:
                previous_capture_time_s = selected_packets[0].capture_time_s
                for packet_index, replay_packet in enumerate(selected_packets, start=1):
                    _sleep_for_capture_delta(previous_capture_time_s, replay_packet.capture_time_s, options)
                    previous_capture_time_s = replay_packet.capture_time_s
                    sent_packets = _send_replay_packet(
                        options=options,
                        replay_packet=replay_packet,
                        loop_index=loop_index,
                        loop_label=loop_label,
                        sent_packets=sent_packets,
                    )
                    last_progress_s = _maybe_print_progress(
                        options=options,
                        loop_index=loop_index,
                        loop_label=loop_label,
                        packet_index=packet_index,
                        packet_count=len(selected_packets),
                        sent_packets=sent_packets,
                        started_at_s=started_at_s,
                        last_progress_s=last_progress_s,
                    )

            should_continue = options.repeat == 0 or loop_index < options.repeat
            if should_continue and options.loop_delay_s > 0:
                time.sleep(options.loop_delay_s)
    except KeyboardInterrupt:
        pass

    return ReplaySummary(
        read_packets=read_packets,
        udp_packets=udp_packets,
        selected_packets=len(selected_packets),
        sent_packets=sent_packets,
    )


def _send_replay_packet(
    *,
    options: ReplayOptions,
    replay_packet: UdpReplayPacket,
    loop_index: int,
    loop_label: str,
    sent_packets: int,
) -> int:
    if options.verbose_packets and not options.quiet:
        action = "would send" if options.dry_run else "send"
        print(
            f"[{loop_index}/{loop_label}] {action} "
            f"{replay_packet.src_ip}:{replay_packet.src_port} -> "
            f"{replay_packet.dst_ip}:{replay_packet.dst_port} "
            f"payload={replay_packet.payload_len} bytes"
        )

    if not options.dry_run:
        send(replay_packet.packet, verbose=False)
    return sent_packets + 1


def _maybe_print_progress(
    *,
    options: ReplayOptions,
    loop_index: int,
    loop_label: str,
    packet_index: int,
    packet_count: int,
    sent_packets: int,
    started_at_s: float,
    last_progress_s: float,
) -> float:
    now_s = time.monotonic()
    should_print_progress = (
        not options.quiet
        and not options.verbose_packets
        and (
            options.progress_interval_s == 0
            or now_s - last_progress_s >= options.progress_interval_s
            or packet_index == packet_count
        )
    )
    if not should_print_progress:
        return last_progress_s
    line = _format_progress_line(
        loop_index=loop_index,
        loop_label=loop_label,
        packet_index=packet_index,
        packet_count=packet_count,
        sent_packets=sent_packets,
        started_at_s=started_at_s,
        dry_run=options.dry_run,
    )
    _print_progress(line, final=packet_index == packet_count)
    return now_s


def parse_args(argv: list[str] | None = None) -> ReplayOptions:
    parser = argparse.ArgumentParser(
        description="Replay UDP packets from a pcap/pcapng file at the IP layer.",
    )
    parser.add_argument(
        "pcap",
        nargs="?",
        default=r"D:\vamdump.pcapng",
        help=r"pcap/pcapng path (default: D:\vamdump.pcapng)",
    )
    parser.add_argument(
        "--dst-ip",
        default="127.0.0.1",
        help="destination IP for replay (default: 127.0.0.1)",
    )
    parser.add_argument("--dst-port", type=_port, default=None, help="override destination UDP port")
    parser.add_argument("--src-ip", default=None, help="override source IP")
    parser.add_argument("--src-port", type=_port, default=None, help="override source UDP port")
    parser.add_argument(
        "--preserve-timing",
        dest="preserve_timing",
        action="store_true",
        default=True,
        help="sleep between packets according to capture timestamps (default)",
    )
    parser.add_argument(
        "--no-preserve-timing",
        dest="preserve_timing",
        action="store_false",
        help="send packets as fast as possible",
    )
    parser.add_argument(
        "--speed",
        type=_positive_float,
        default=1.0,
        help="timing multiplier when --preserve-timing is set (default: 1)",
    )
    parser.add_argument(
        "--fixed-rate-hz",
        type=_non_negative_float,
        default=0.0,
        help=(
            "coalesce capture-time windows and replay non-empty windows at this fixed rate; "
            "0 disables fixed-rate replay (default: 0)"
        ),
    )
    parser.add_argument(
        "--max-delay-s",
        type=_non_negative_float,
        default=0.0,
        help="cap per-packet sleep when preserving timing; 0 disables cap (default: 0)",
    )
    parser.add_argument(
        "--repeat",
        type=_repeat_count,
        default=1,
        help="number of replay loops; 0 means forever (default: 1)",
    )
    parser.add_argument(
        "--loop-delay-s",
        type=_non_negative_float,
        default=0.0,
        help="sleep between repeat loops (default: 0)",
    )
    parser.add_argument("--limit", type=_optional_limit, default=None, help="maximum selected UDP packets to replay")
    parser.add_argument(
        "--dedupe-window-ms",
        type=_non_negative_float,
        default=0.0,
        help="drop identical UDP packets seen within this capture-time window (default: 0)",
    )
    parser.add_argument(
        "--progress-interval-s",
        type=_non_negative_float,
        default=0.5,
        help="progress refresh interval; 0 refreshes every packet (default: 0.5)",
    )
    parser.add_argument("--verbose-packets", action="store_true", help="print one log line per selected UDP packet")
    parser.add_argument("--dry-run", action="store_true", help="print selected packets without sending")
    parser.add_argument("--quiet", action="store_true", help="only print the final summary")
    args = parser.parse_args(argv)

    return ReplayOptions(
        pcap_path=Path(str(args.pcap)).expanduser(),
        dst_ip=str(args.dst_ip),
        dst_port=args.dst_port,
        src_ip=args.src_ip,
        src_port=args.src_port,
        preserve_timing=bool(args.preserve_timing),
        fixed_rate_hz=float(args.fixed_rate_hz),
        speed=float(args.speed),
        max_delay_s=float(args.max_delay_s),
        repeat=int(args.repeat),
        loop_delay_s=float(args.loop_delay_s),
        limit=args.limit,
        dedupe_window_s=float(args.dedupe_window_ms) / 1000.0,
        progress_interval_s=float(args.progress_interval_s),
        verbose_packets=bool(args.verbose_packets),
        dry_run=bool(args.dry_run),
        quiet=bool(args.quiet),
    )


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    try:
        summary = replay_udp_pcap(options)
    except OSError as exc:
        raise SystemExit(f"Failed to replay pcap: {exc}") from exc

    print(
        "Done: "
        f"read={summary.read_packets} "
        f"udp={summary.udp_packets} "
        f"selected={summary.selected_packets} "
        f"{'dry_run=' if options.dry_run else 'sent='}{summary.sent_packets}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
