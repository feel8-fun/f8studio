#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "f8cppsdk/rungraph_routes.h"

using json = nlohmann::json;

TEST(RungraphRoutes, ParsesCrossServiceDataEdgesForTargetService) {
  json g;
  g["nodes"] = json::array();
  g["edges"] = json::array({
      // cross-service data edge into svcB
      json{{"edgeId", "e1"},
           {"kind", "data"},
           {"fromServiceId", "svcA"},
           {"fromOperatorId", "op1"},
           {"fromPort", "out"},
           {"toServiceId", "svcB"},
           {"toOperatorId", "op2"},
           {"toPort", "in"}},
      // intra edge should be ignored
      json{{"edgeId", "e2"},
           {"kind", "data"},
           {"fromServiceId", "svcB"},
           {"fromOperatorId", "x"},
           {"fromPort", "o"},
           {"toServiceId", "svcB"},
           {"toOperatorId", "y"},
           {"toPort", "i"}},
      // non-data kind ignored
      json{{"edgeId", "e3"},
           {"kind", "state"},
           {"fromServiceId", "svcA"},
           {"fromOperatorId", "op1"},
           {"fromPort", "foo"},
           {"toServiceId", "svcB"},
           {"toOperatorId", "op2"},
           {"toPort", "bar"}},
  });

  const auto routes = f8::cppsdk::parse_cross_service_data_routes(g, "svcB");
  ASSERT_EQ(routes.size(), 1u);

  const auto& kv = *routes.begin();
  EXPECT_EQ(kv.first, "svc.svcA.nodes.op1.data.out");
  ASSERT_EQ(kv.second.size(), 1u);
  EXPECT_EQ(kv.second[0].to_node_id, "op2");
  EXPECT_EQ(kv.second[0].to_port, "in");
  EXPECT_EQ(kv.second[0].from_service_id, "svcA");
  EXPECT_EQ(kv.second[0].from_node_id, "op1");
  EXPECT_EQ(kv.second[0].from_port, "out");
}

TEST(RungraphRoutes, ServiceNodeDefaultsOperatorIdToServiceId) {
  json g;
  g["edges"] = json::array(
      {json{{"edgeId", "e1"},
            {"kind", "data"},
            {"fromServiceId", "svcA"},
            {"fromOperatorId", nullptr},
            {"fromPort", "out"},
            {"toServiceId", "svcB"},
            {"toOperatorId", nullptr},
            {"toPort", "in"}}});

  const auto routes = f8::cppsdk::parse_cross_service_data_routes(g, "svcB");
  ASSERT_EQ(routes.size(), 1u);
  const auto& kv = *routes.begin();
  EXPECT_EQ(kv.first, "svc.svcA.nodes.svcA.data.out");
  ASSERT_EQ(kv.second.size(), 1u);
  EXPECT_EQ(kv.second[0].to_node_id, "svcB");
}
