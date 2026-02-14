#include <iostream>
#include <string>

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>

#include "f8cppsdk/describe_builtins.h"

int main(int argc, char** argv) {
  cxxopts::Options options("f8dlinfer_detector_service", "Experimental DL inference C++ service skeleton");
  options.add_options()
      ("describe", "Print service description")
      ("service-id", "Service instance id", cxxopts::value<std::string>()->default_value(""))
      ("nats-url", "NATS URL", cxxopts::value<std::string>()->default_value("nats://127.0.0.1:4222"))
      ("h,help", "Show help");

  const auto parsed = options.parse(argc, argv);
  if (parsed.count("help") > 0U) {
    std::cout << options.help() << "\n";
    return 0;
  }

  if (parsed.count("describe") > 0U) {
    nlohmann::json svc;
    svc["schemaVersion"] = "f8service/1";
    svc["serviceClass"] = "f8.dl.detector.cpp";
    svc["version"] = "0.0.1";
    svc["label"] = "DL Detector (C++ Experimental)";
    svc["description"] = "Experimental C++ detector shell. TensorRT backend is not wired yet.";
    svc["tags"] = nlohmann::json::array({"cpp", "dl", "experimental"});
    svc["rendererClass"] = "default_svc";
    svc["stateFields"] = nlohmann::json::array();
    svc["dataInPorts"] = nlohmann::json::array();
    svc["dataOutPorts"] = nlohmann::json::array(
        {nlohmann::json{{"name", "detections"}, {"valueSchema", nlohmann::json{{"type", "any"}}}}});

    nlohmann::json payload;
    payload["schemaVersion"] = "f8describe/1";
    payload["service"] = svc;
    payload["operators"] = nlohmann::json::array();
    const nlohmann::json normalized = f8::cppsdk::normalize_describe_with_builtin_state_fields(payload);
    std::cout << normalized.dump(1) << "\n";
    return 0;
  }

  const std::string service_id = parsed["service-id"].as<std::string>();
  if (service_id.empty()) {
    std::cerr << "Missing --service-id\n";
    return 2;
  }
  std::cerr << "f8dlinfer_detector_service is a Phase-B skeleton only. service-id=" << service_id << "\n";
  return 3;
}
