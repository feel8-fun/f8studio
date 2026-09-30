#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "f8cppsdk/exec_flow_executor.h"
#include "f8cppsdk/runtime_node_registry.h"
#include "f8cppsdk/service_host.h"

using json = nlohmann::json;

namespace {

class RecordingNode final : public f8::cppsdk::OperatorNode,
                            public f8::cppsdk::ComputableNode,
                            public f8::cppsdk::EntrypointNode {
 public:
  RecordingNode(std::string node_id, std::vector<std::string> route_ports, std::vector<std::string>* calls)
      : OperatorNode(std::move(node_id), {"in"}, {"out"}, {}, {"exec"}, std::move(route_ports)), calls_(calls) {}

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)exec_id;
    std::lock_guard<std::mutex> lock(mu_);
    calls_->push_back(node_id() + "." + in_port);
    return exec_out_ports();
  }

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    (void)ctx_id;
    if (port == "out") return json{{"node", node_id()}};
    return nullptr;
  }

  void start_entrypoint(const f8::cppsdk::EntrypointContext& ctx) override { (void)ctx; }

  void stop_entrypoint() override {}

 private:
  std::vector<std::string>* calls_;
  std::mutex mu_;
};

class ConstantNode final : public f8::cppsdk::OperatorNode, public f8::cppsdk::ComputableNode {
 public:
  ConstantNode(std::string node_id, json value)
      : OperatorNode(std::move(node_id), {}, {"out"}, {}, {}, {}), value_(std::move(value)) {}

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    last_ctx_id = ctx_id;
    if (port != "out") return nullptr;
    return value_;
  }

  std::int64_t last_ctx_id = 0;

 private:
  json value_;
};

class AddOneNode final : public f8::cppsdk::OperatorNode, public f8::cppsdk::ComputableNode {
 public:
  explicit AddOneNode(std::string node_id)
      : OperatorNode(std::move(node_id), {"in"}, {"out"}, {}, {}, {}) {}

  json compute_output(const std::string& port, std::int64_t ctx_id) override {
    if (port != "out") return nullptr;
    const auto raw = pull("in", ctx_id);
    if (!raw.has_value() || !raw->is_number()) return nullptr;
    return raw->get<double>() + 1.0;
  }
};

class PullingSinkNode final : public f8::cppsdk::OperatorNode {
 public:
  explicit PullingSinkNode(std::string node_id)
      : OperatorNode(std::move(node_id), {"in"}, {}, {}, {"exec"}, {}) {}

  std::vector<std::string> on_exec(std::int64_t exec_id, const std::string& in_port) override {
    (void)in_port;
    last_value = pull("in", exec_id).value_or(nullptr);
    last_exec_id = exec_id;
    return {};
  }

  json last_value = nullptr;
  std::int64_t last_exec_id = 0;
};

json runtime_node(const std::string& service_id, const std::string& node_id, const std::string& op,
                  json exec_in = json::array({"exec"}), json exec_out = json::array({"exec"})) {
  return json{{"nodeId", node_id},
              {"serviceId", service_id},
              {"serviceClass", "f8.test"},
              {"operatorClass", op},
              {"execInPorts", exec_in},
              {"execOutPorts", exec_out},
              {"dataInPorts", json::array({json{{"name", "in"}, {"payload", {{"kind", "json"}, {"valueSchema", json{{"type", "any"}}}}}}})},
              {"dataOutPorts", json::array({json{{"name", "out"}, {"payload", {{"kind", "json"}, {"valueSchema", json{{"type", "any"}}}}}}})},
              {"stateFields", json::array()}};
}

json exec_edge(const std::string& id, const std::string& service_id, const std::string& from_node,
               const std::string& from_port, const std::string& to_node, const std::string& to_port) {
  return json{{"edgeId", id},
              {"kind", "exec"},
              {"fromServiceId", service_id},
              {"fromOperatorId", from_node},
              {"fromPort", from_port},
              {"toServiceId", service_id},
              {"toOperatorId", to_node},
              {"toPort", to_port}};
}

json data_edge(const std::string& id, const std::string& service_id, const std::string& from_node,
               const std::string& from_port, const std::string& to_node, const std::string& to_port) {
  return json{{"edgeId", id},
              {"kind", "data"},
              {"fromServiceId", service_id},
              {"fromOperatorId", from_node},
              {"fromPort", from_port},
              {"toServiceId", service_id},
              {"toOperatorId", to_node},
              {"toPort", to_port}};
}

json data_half_edge(const std::string& id, const std::string& service_id, const std::string& from_node,
                    const std::string& from_port) {
  return json{{"edgeId", id},
              {"kind", "data"},
              {"direction", "out"},
              {"fromServiceId", service_id},
              {"fromOperatorId", from_node},
              {"fromPort", from_port},
              {"toServiceId", "remote"},
              {"toOperatorId", "remoteNode"},
              {"toPort", "in"}};
}

f8::cppsdk::generated::F8RuntimeGraph parse_graph(const json& graph) {
  f8::cppsdk::generated::F8RuntimeGraph out;
  f8::cppsdk::generated::ParseError err;
  EXPECT_TRUE(f8::cppsdk::generated::parse_F8RuntimeGraph(graph, out, err)) << err.message;
  return out;
}

}  // namespace

TEST(CppExecFlow, RejectsDuplicateExecOut) {
  json graph{{"graphId", "g"}, {"revision", "r"}, {"nodes", json::array()}, {"edges", json::array()}};
  graph["edges"].push_back(exec_edge("e1", "svc", "a", "exec", "b", "exec"));
  graph["edges"].push_back(exec_edge("e2", "svc", "a", "exec", "c", "exec"));
  EXPECT_THROW((void)f8::cppsdk::validate_exec_topology_or_throw(parse_graph(graph), "svc"), std::invalid_argument);
}

TEST(CppExecFlow, RejectsDuplicateExecIn) {
  json graph{{"graphId", "g"}, {"revision", "r"}, {"nodes", json::array()}, {"edges", json::array()}};
  graph["edges"].push_back(exec_edge("e1", "svc", "a", "exec", "c", "exec"));
  graph["edges"].push_back(exec_edge("e2", "svc", "b", "exec", "c", "exec"));
  EXPECT_THROW((void)f8::cppsdk::validate_exec_topology_or_throw(parse_graph(graph), "svc"), std::invalid_argument);
}

TEST(CppExecFlow, RejectsCycles) {
  json graph{{"graphId", "g"}, {"revision", "r"}, {"nodes", json::array()}, {"edges", json::array()}};
  graph["edges"].push_back(exec_edge("e1", "svc", "a", "exec", "b", "exec"));
  graph["edges"].push_back(exec_edge("e2", "svc", "b", "exec", "a", "exec"));
  EXPECT_THROW((void)f8::cppsdk::validate_exec_topology_or_throw(parse_graph(graph), "svc"), std::invalid_argument);
}

TEST(CppExecFlow, PropagatesDepthFirstAndEmitsHalfEdgeOutputs) {
  f8::cppsdk::ServiceBus::Config cfg;
  cfg.service_id = "svc";
  cfg.service_class = "f8.test";
  cfg.bus_backend = f8::cppsdk::BusBackend::kMem;
  f8::cppsdk::ServiceBus bus(cfg);
  f8::cppsdk::ExecFlowExecutor executor(bus);

  std::vector<std::string> calls;
  RecordingNode start("start", {"a"}, &calls);
  RecordingNode a("a", {"b"}, &calls);
  RecordingNode b("b", {}, &calls);
  start.attach(&bus);
  a.attach(&bus);
  b.attach(&bus);
  executor.register_node(&start);
  executor.register_node(&a);
  executor.register_node(&b);

  json graph{{"graphId", "g"},
             {"revision", "r"},
             {"nodes", json::array({runtime_node("svc", "start", "op", json::array(), json::array({"a"})),
                                     runtime_node("svc", "a", "op", json::array({"exec"}), json::array({"b"})),
                                     runtime_node("svc", "b", "op", json::array({"exec"}), json::array())})},
             {"edges", json::array({exec_edge("e1", "svc", "start", "a", "a", "exec"),
                                     exec_edge("e2", "svc", "a", "b", "b", "exec"),
                                     data_half_edge("d1", "svc", "a", "out")})}};
  executor.apply_rungraph(graph);
  executor.trigger_exec("start", "a", 1);

  ASSERT_EQ(calls.size(), 2u);
  EXPECT_EQ(calls[0], "a.exec");
  EXPECT_EQ(calls[1], "b.exec");
}

TEST(CppExecFlow, PullResolvesLocalComputableDataEdge) {
  f8::cppsdk::ServiceBus::Config cfg;
  cfg.service_id = "svc";
  cfg.service_class = "f8.test";
  cfg.bus_backend = f8::cppsdk::BusBackend::kMem;
  f8::cppsdk::ServiceBus bus(cfg);
  f8::cppsdk::ExecFlowExecutor executor(bus);

  std::vector<std::string> calls;
  RecordingNode start("start", {"exec"}, &calls);
  ConstantNode source("source", 42.0);
  PullingSinkNode sink("sink");
  start.attach(&bus);
  source.attach(&bus);
  sink.attach(&bus);
  executor.register_node(&start);
  executor.register_node(&source);
  executor.register_node(&sink);

  json graph{{"graphId", "g"},
             {"revision", "r"},
             {"nodes", json::array({runtime_node("svc", "start", "op", json::array(), json::array({"exec"})),
                                     runtime_node("svc", "source", "op", json::array(), json::array()),
                                     runtime_node("svc", "sink", "op", json::array({"exec"}), json::array())})},
             {"edges", json::array({exec_edge("e1", "svc", "start", "exec", "sink", "exec"),
                                     data_edge("d1", "svc", "source", "out", "sink", "in")})}};
  executor.apply_rungraph(graph);
  executor.trigger_exec("start", "exec", 123);

  EXPECT_EQ(sink.last_exec_id, 123);
  EXPECT_EQ(sink.last_value, json(42.0));
  EXPECT_EQ(source.last_ctx_id, 123);
}

TEST(CppExecFlow, PullResolvesChainedLocalComputableDataEdges) {
  f8::cppsdk::ServiceBus::Config cfg;
  cfg.service_id = "svc";
  cfg.service_class = "f8.test";
  cfg.bus_backend = f8::cppsdk::BusBackend::kMem;
  f8::cppsdk::ServiceBus bus(cfg);
  f8::cppsdk::ExecFlowExecutor executor(bus);

  std::vector<std::string> calls;
  RecordingNode start("start", {"exec"}, &calls);
  ConstantNode source("source", 10.0);
  AddOneNode mid("mid");
  PullingSinkNode sink("sink");
  start.attach(&bus);
  source.attach(&bus);
  mid.attach(&bus);
  sink.attach(&bus);
  executor.register_node(&start);
  executor.register_node(&source);
  executor.register_node(&mid);
  executor.register_node(&sink);

  json graph{{"graphId", "g"},
             {"revision", "r"},
             {"nodes", json::array({runtime_node("svc", "start", "op", json::array(), json::array({"exec"})),
                                     runtime_node("svc", "source", "op", json::array(), json::array()),
                                     runtime_node("svc", "mid", "op", json::array(), json::array()),
                                     runtime_node("svc", "sink", "op", json::array({"exec"}), json::array())})},
             {"edges", json::array({exec_edge("e1", "svc", "start", "exec", "sink", "exec"),
                                     data_edge("d1", "svc", "source", "out", "mid", "in"),
                                     data_edge("d2", "svc", "mid", "out", "sink", "in")})}};
  executor.apply_rungraph(graph);
  executor.trigger_exec("start", "exec", 124);

  EXPECT_EQ(sink.last_exec_id, 124);
  EXPECT_EQ(sink.last_value, json(11.0));
  EXPECT_EQ(source.last_ctx_id, 124);
}

TEST(CppServiceHost, CreatesRecreatesAndRemovesNodes) {
  f8::cppsdk::ServiceBus::Config cfg;
  cfg.service_id = "svc";
  cfg.service_class = "f8.test";
  cfg.bus_backend = f8::cppsdk::BusBackend::kMem;
  f8::cppsdk::ServiceBus bus(cfg);

  f8::cppsdk::RuntimeNodeRegistry registry;
  registry.register_service_spec(json{{"specKind", "service"}, {"serviceClass", "f8.test"}, {"label", "Test"}}, true);
  registry.register_operator_spec(json{{"specKind", "operator"},
                                       {"serviceClass", "f8.test"},
                                       {"operatorClass", "op"},
                                       {"label", "Op"}},
                                  true);
  std::vector<std::string> calls;
  registry.register_operator_factory(
      "f8.test", "op",
      [&calls](const std::string& node_id, const f8::cppsdk::generated::F8RuntimeNode& node, const json& initial_state) {
        (void)node;
        (void)initial_state;
        return std::make_unique<RecordingNode>(node_id, std::vector<std::string>{"exec"}, &calls);
      },
      true);

  f8::cppsdk::ServiceHost host(bus, registry, "f8.test");
  host.start();

  json graph{{"graphId", "g"}, {"revision", "r"}, {"nodes", json::array({runtime_node("svc", "n1", "op")})}, {"edges", json::array()}};
  std::string code;
  std::string message;
  ASSERT_TRUE(host.apply_rungraph(graph, code, message)) << message;
  ASSERT_NE(host.get_node("n1"), nullptr);

  graph["nodes"][0]["dataInPorts"].push_back(json{{"name", "extra"}, {"payload", {{"kind", "json"}, {"valueSchema", json{{"type", "any"}}}}}});
  ASSERT_TRUE(host.apply_rungraph(graph, code, message)) << message;
  ASSERT_NE(host.get_node("n1"), nullptr);

  graph["nodes"] = json::array();
  ASSERT_TRUE(host.apply_rungraph(graph, code, message)) << message;
  EXPECT_EQ(host.get_node("n1"), nullptr);
}
