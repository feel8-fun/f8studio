#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "f8cppsdk/generated/protocol_models.h"

namespace {

using json = nlohmann::json;
using namespace f8::cppsdk::generated;

TEST(ProtocolModelsParse, CommandInvoke_IgnoreExtra) {
  json j = json::object();
  j["call"] = "pickRegion";
  j["args"] = json::object();
  j["args"]["x"] = 1;
  j["meta"] = json::object();
  j["meta"]["traceId"] = "t1";
  j["extra"] = 123;  // should be ignored

  F8CommandInvokeRequest req{};
  ParseError err{};
  EXPECT_TRUE(parse_F8CommandInvokeRequest(j, req, err)) << err.message;
  EXPECT_EQ(req.call, "pickRegion");
  EXPECT_TRUE(req.args.is_object());
  EXPECT_TRUE(req.meta.is_object());
}

TEST(ProtocolModelsParse, SetActiveArgs_Parse) {
  json j = json::object();
  j["active"] = true;
  j["unexpected"] = "ok";

  F8SetActiveArgs req{};
  ParseError err{};
  EXPECT_TRUE(parse_F8SetActiveArgs(j, req, err)) << err.message;
  EXPECT_TRUE(req.active);
}

TEST(ProtocolModelsParse, SetStateArgs_RequiresValue) {
  {
    json j = json::object();
    j["nodeId"] = "svc.demo";
    j["field"] = "active";

    F8SetStateArgs req{};
    ParseError err{};
    EXPECT_FALSE(parse_F8SetStateArgs(j, req, err));
  }
  {
    json j = json::object();
    j["nodeId"] = "svc.demo";
    j["field"] = "active";
    j["value"] = nullptr;  // allowed
    j["extra"] = 1;

    F8SetStateArgs req{};
    ParseError err{};
    EXPECT_TRUE(parse_F8SetStateArgs(j, req, err)) << err.message;
    EXPECT_EQ(req.nodeId, "svc.demo");
    EXPECT_EQ(req.field, "active");
    EXPECT_TRUE(req.value.is_null());
  }
}

TEST(ProtocolModelsParse, SetRungraphArgs_Parse) {
  json graph = json::object();
  graph["graphId"] = "g1";
  graph["revision"] = "r1";

  json j = json::object();
  j["graph"] = graph;
  j["extra"] = "ignored";

  F8SetRungraphArgs req{};
  ParseError err{};
  EXPECT_TRUE(parse_F8SetRungraphArgs(j, req, err)) << err.message;
  EXPECT_EQ(req.graph.graphId, "g1");
  EXPECT_EQ(req.graph.revision, "r1");
}

}  // namespace

