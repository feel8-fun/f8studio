#pragma once

#include <nlohmann/json.hpp>

namespace f8::cppsdk {

// Normalize describe payload by force-applying SDK builtin state fields.
//
// Supported input shapes:
// - f8describe/1 payload: {"service": {...}, "operators": [...]}
// - f8service/1 payload:  {"serviceClass": "...", ...}
//
// Builtins:
// - service:  active(rw,bool,default=true,showOnNode=true), svcId(ro,string,showOnNode=false)
// - operator: svcId(ro,string,showOnNode=false), operatorId(ro,string,showOnNode=false)
nlohmann::json normalize_describe_with_builtin_state_fields(const nlohmann::json& payload);

}  // namespace f8::cppsdk
