#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "f8cppsdk/service_bus.h"
#include "f8cppsdk/time_utils.h"

using json = nlohmann::json;

namespace {

std::string unique_token(const std::string& prefix) {
  static std::atomic<std::uint64_t> seq{0};
  return prefix + std::to_string(f8::cppsdk::now_ms()) + "_" + std::to_string(seq.fetch_add(1));
}

f8::cppsdk::ServiceBus::Config zenoh_config(const std::string& service_id) {
  f8::cppsdk::ServiceBus::Config cfg;
  cfg.service_id = service_id;
  cfg.bus_backend = f8::cppsdk::BusBackend::kZenoh;
  cfg.monitor_enabled = false;
  return cfg;
}

bool wait_until(const std::function<bool()>& pred, const std::function<void()>& pump, std::int64_t timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    pump();
    if (pred()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  pump();
  return pred();
}

class BusStopper final {
 public:
  void add(f8::cppsdk::ServiceBus& bus) { buses_[count_++] = &bus; }

  ~BusStopper() {
    for (std::size_t i = 0; i < count_; ++i) {
      if (buses_[i] != nullptr) {
        buses_[i]->stop();
      }
    }
  }

 private:
  f8::cppsdk::ServiceBus* buses_[4] = {};
  std::size_t count_ = 0;
};

class RecordingDataNode final : public f8::cppsdk::DataReceivableNode {
 public:
  void on_data(const std::string& node_id, const std::string& port, const json& value, std::int64_t ts_ms,
               const json& meta) override {
    last_node_id = node_id;
    last_port = port;
    last_value = value;
    last_ts_ms = ts_ms;
    last_meta = meta;
    ++count;
  }

  int count = 0;
  std::string last_node_id;
  std::string last_port;
  json last_value = json::object();
  std::int64_t last_ts_ms = 0;
  json last_meta = json::object();
};

class RecordingStateNode final : public f8::cppsdk::StatefulNode {
 public:
  void on_state(const std::string& node_id, const std::string& field, const json& value, std::int64_t ts_ms,
                const json& meta) override {
    last_node_id = node_id;
    last_field = field;
    last_value = value;
    last_ts_ms = ts_ms;
    last_meta = meta;
    ++count;
  }

  int count = 0;
  std::string last_node_id;
  std::string last_field;
  json last_value = json(nullptr);
  std::int64_t last_ts_ms = 0;
  json last_meta = json::object();
};

std::uint32_t fnv1a32(const std::string& text) {
  std::uint32_t value = 0x811C9DC5u;
  for (unsigned char ch : text) {
    value ^= static_cast<std::uint32_t>(ch);
    value *= 0x01000193u;
  }
  return value;
}

std::string command_key_for_name(const std::string& name) {
  std::string base;
  bool last_was_sep = false;
  for (unsigned char ch : name) {
    const char lower = static_cast<char>(std::tolower(ch));
    if ((lower >= 'a' && lower <= 'z') || (lower >= '0' && lower <= '9')) {
      base.push_back(lower);
      last_was_sep = false;
      continue;
    }
    if (!last_was_sep) {
      base.push_back('_');
      last_was_sep = true;
    }
  }
  while (!base.empty() && base.front() == '_') base.erase(base.begin());
  while (!base.empty() && base.back() == '_') base.pop_back();
  if (base.empty()) base = "command";
  std::ostringstream out;
  out << base << "_" << std::hex << std::nouppercase << std::setw(8) << std::setfill('0') << fnv1a32(name);
  return out.str();
}

std::string command_input_state_field(const std::string& name) {
  return "__cmd__." + command_key_for_name(name) + ".in";
}

std::string command_output_state_field(const std::string& name) {
  return "__cmd__." + command_key_for_name(name) + ".out";
}

class RecordingCommandNode final : public f8::cppsdk::CommandableNode {
 public:
  bool on_command(const std::string& call, const json& args, const json& meta, json& result, std::string& error_code,
                  std::string& error_message) override {
    last_call = call;
    last_args = args;
    last_meta = meta;
    ++count;
    error_code.clear();
    error_message.clear();
    result = json{{"call", call}, {"args", args}};
    return true;
  }

  int count = 0;
  std::string last_call;
  json last_args = json::object();
  json last_meta = json::object();
};

class RejectingSetStateNode final : public f8::cppsdk::SetStateHandlerNode {
 public:
  bool on_set_state(const std::string& node_id, const std::string& field, const json& value, const json& meta,
                    std::string& error_code, std::string& error_message) override {
    last_node_id = node_id;
    last_field = field;
    last_value = value;
    last_meta = meta;
    ++count;
    error_code = "UNKNOWN_FIELD";
    error_message = "unknown state field";
    return false;
  }

  int count = 0;
  std::string last_node_id;
  std::string last_field;
  json last_value = json(nullptr);
  json last_meta = json::object();
};

json service_node(const std::string& service_id, const std::string& node_id, const json& state_fields = json::array()) {
  json node{{"nodeId", node_id}, {"serviceId", service_id}, {"serviceClass", "demo"}, {"stateFields", state_fields}};
  if (node_id != service_id) node["operatorClass"]="OpClass";
  return node;
}

}  // namespace

TEST(ServiceBusIntegration, ZenohDataRouteDeliversAndBuffers) {
  const std::string svc_a = unique_token("svcA_");
  const std::string svc_b = unique_token("svcB_");

  f8::cppsdk::ServiceBus bus_a(zenoh_config(svc_a));
  f8::cppsdk::ServiceBus bus_b(zenoh_config(svc_b));
  RecordingDataNode data_node;
  bus_b.add_data_node(&data_node);

  if (!bus_a.start()) GTEST_SKIP() << "Zenoh runtime unavailable";
  BusStopper stopper;
  stopper.add(bus_a);
  if (!bus_b.start()) GTEST_SKIP() << "Zenoh runtime unavailable";
  stopper.add(bus_b);

  json graph;
  graph["graphId"] = "g1";
  graph["revision"] = "r1";
  graph["nodes"] = json::array({
      service_node(svc_a, "op1"),
      service_node(svc_b, "op2"),
  });
  graph["edges"] = json::array({
      json{{"edgeId", "e1"},
           {"kind", "data"},
           {"fromServiceId", svc_a},
           {"fromOperatorId", "op1"},
           {"fromPort", "out"},
           {"toServiceId", svc_b},
           {"toOperatorId", "op2"},
           {"toPort", "in"},
           {"strategy", "latest"},
           {"timeoutMs", 0}},
  });

  std::string err_code;
  std::string err_msg;
  ASSERT_TRUE(bus_b.on_set_rungraph(graph, json::object(), err_code, err_msg)) << err_code << ": " << err_msg;

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_TRUE(bus_a.emit_data("op1", "out", json{{"x", 1}}, f8::cppsdk::now_ms()));

  ASSERT_TRUE(wait_until([&]() { return data_node.count > 0; }, [&]() { (void)bus_b.drain_main_thread(); }, 3000))
      << "data route did not deliver";
  EXPECT_EQ(data_node.last_node_id, "op2");
  EXPECT_EQ(data_node.last_port, "in");
  EXPECT_EQ(data_node.last_value.value("x", 0), 1);

  auto pulled = bus_b.pull_data("op2", "in");
  ASSERT_TRUE(pulled.has_value());
  EXPECT_EQ(pulled->value("x", 0), 1);
}

TEST(ServiceBusIntegration, ZenohRetainedStateRouteMirrorsRemoteState) {
  const std::string svc_a = unique_token("svcA_");
  const std::string svc_b = unique_token("svcB_");

  f8::cppsdk::ServiceBus bus_a(zenoh_config(svc_a));
  f8::cppsdk::ServiceBus bus_b(zenoh_config(svc_b));
  RecordingStateNode state_node;
  bus_b.add_stateful_node(&state_node);

  if (!bus_a.start()) GTEST_SKIP() << "Zenoh runtime unavailable";
  BusStopper stopper;
  stopper.add(bus_a);
  if (!bus_b.start()) GTEST_SKIP() << "Zenoh runtime unavailable";
  stopper.add(bus_b);

  json graph;
  graph["graphId"] = "g1";
  graph["revision"] = "r1";
  graph["nodes"] = json::array({
      service_node(svc_a, "op1", json::array({json{{"name", "out"}, {"access", "rw"}, {"valueSchema", json{{"type", "any"}}}}})),
      service_node(svc_b, "op2", json::array({json{{"name", "in"}, {"access", "rw"}, {"valueSchema", json{{"type", "any"}}}}})),
  });
  graph["edges"] = json::array({
      json{{"edgeId", "e1"},
           {"kind", "state"},
           {"fromServiceId", svc_a},
           {"fromOperatorId", "op1"},
           {"fromPort", "out"},
           {"toServiceId", svc_b},
           {"toOperatorId", "op2"},
           {"toPort", "in"},
           {"strategy", "latest"}},
  });

  std::string err_code;
  std::string err_msg;
  ASSERT_TRUE(bus_b.on_set_rungraph(graph, json::object(), err_code, err_msg)) << err_code << ": " << err_msg;

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ASSERT_TRUE(bus_a.publish_state("op1", "out", "v1", "runtime", json::object(), f8::cppsdk::now_ms(), "runtime"));

  ASSERT_TRUE(wait_until([&]() { return state_node.count > 0; }, [&]() { (void)bus_b.drain_main_thread(); }, 3000))
      << "state route did not deliver";
  EXPECT_EQ(state_node.last_node_id, "op2");
  EXPECT_EQ(state_node.last_field, "in");
  EXPECT_EQ(state_node.last_value, json("v1"));

  const auto mirrored = bus_b.get_state("op2", "in");
  ASSERT_TRUE(mirrored.found);
  EXPECT_EQ(mirrored.value, json("v1"));
}

TEST(ServiceBusIntegration, HiddenCommandInputBypassesRejectingSetStateHandler) {
  const std::string svc = unique_token("svcB_");
  f8::cppsdk::ServiceBus bus(zenoh_config(svc));
  RecordingCommandNode command_node;
  RecordingStateNode state_node;
  RejectingSetStateNode rejecting_state_node;
  bus.add_command_node(
      &command_node,
      json{{"service",
            {{"commands",
              json::array({
                  json{{"name", "run"}, {"params", json::array({json{{"name", "a"}}, json{{"name", "b"}}})}},
              })}}}});
  bus.add_stateful_node(&state_node);
  bus.add_set_state_node(&rejecting_state_node);

  if (!bus.start()) GTEST_SKIP() << "Zenoh runtime unavailable";
  BusStopper stopper;
  stopper.add(bus);

  const std::string input_field = command_input_state_field("run");
  const std::string output_field = command_output_state_field("run");

  json graph;
  graph["graphId"] = "g1";
  graph["revision"] = "r1";
  graph["nodes"] = json::array({
      service_node(
          svc,
          svc,
          json::array({
              json{{"name", input_field}, {"access", "wo"}, {"valueSchema", json{{"type", "any"}}}},
              json{{"name", output_field}, {"access", "ro"}, {"valueSchema", json{{"type", "any"}}}},
              json{{"name", "result"}, {"access", "rw"}, {"valueSchema", json{{"type", "any"}}}},
          })),
  });
  graph["edges"] = json::array({
      json{{"edgeId", "e1"},
           {"kind", "state"},
           {"fromServiceId", svc},
           {"fromOperatorId", svc},
           {"fromPort", output_field},
           {"toServiceId", svc},
           {"toOperatorId", svc},
           {"toPort", "result"},
           {"strategy", "latest"}},
  });

  std::string err_code;
  std::string err_msg;
  ASSERT_TRUE(bus.on_set_rungraph(graph, json::object(), err_code, err_msg)) << err_code << ": " << err_msg;

  ASSERT_TRUE(bus.on_set_state(svc, input_field, json::array({1, 2, 3}), json::object(), err_code, err_msg))
      << err_code << ": " << err_msg;

  ASSERT_TRUE(wait_until([&]() { return command_node.count > 0; }, [&]() { (void)bus.drain_main_thread(); }, 3000))
      << "command input did not dispatch";
  ASSERT_TRUE(wait_until([&]() { return state_node.count > 0; }, [&]() { (void)bus.drain_main_thread(); }, 3000))
      << "command output did not fan out";

  EXPECT_EQ(rejecting_state_node.count, 0);
  EXPECT_EQ(command_node.last_call, "run");
  ASSERT_TRUE(command_node.last_args.is_object());
  EXPECT_EQ(command_node.last_args.value("a", 0), 1);
  EXPECT_EQ(command_node.last_args.value("b", 0), 2);
  EXPECT_FALSE(command_node.last_args.contains("c"));

  EXPECT_EQ(state_node.last_node_id, svc);
  EXPECT_EQ(state_node.last_field, "result");
  ASSERT_TRUE(state_node.last_value.is_object());
  EXPECT_EQ(state_node.last_value["call"], "run");
}
