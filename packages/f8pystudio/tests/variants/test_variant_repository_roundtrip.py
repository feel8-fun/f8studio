import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PKG_STUDIO not in sys.path:
    sys.path.insert(0, PKG_STUDIO)


class VariantRepositoryRoundtripTests(unittest.TestCase):
    def test_roundtrip_import_export(self) -> None:
        from f8pystudio.variants.variant_models import F8NodeVariantLibraryFile, F8NodeVariantRecord, F8VariantKind
        from f8pystudio.variants.variant_repository import (
            delete_variant,
            export_to_json,
            import_from_json,
            list_variants_for_base,
            load_library,
            save_library,
            upsert_variant,
        )

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            with patch("pathlib.Path.home", return_value=home):
                rec = F8NodeVariantRecord(
                    variantId="v1",
                    kind=F8VariantKind.operator,
                    baseNodeType="svc.base.node",
                    serviceClass="svc.base",
                    operatorClass="f8.base",
                    name="Variant 1",
                    description="desc",
                    tags=["a", "b"],
                    spec={"schemaVersion": "f8operator/1", "serviceClass": "svc.base", "operatorClass": "f8.base"},
                    createdAt=F8NodeVariantRecord.now_iso(),
                    updatedAt=F8NodeVariantRecord.now_iso(),
                )
                save_library(F8NodeVariantLibraryFile(variants=[rec]))
                lib = load_library()
                self.assertEqual(len(lib.variants), 1)
                self.assertEqual(lib.variants[0].variantId, "v1")

                rec2 = rec.model_copy(deep=True)
                rec2.name = "Variant 2"
                rec2.variantId = "v2"
                upsert_variant(rec2)
                self.assertEqual(len(list_variants_for_base("svc.base.node")), 2)

                out = export_to_json(str(home / "out.json"))
                self.assertTrue(out.is_file())

                delete_variant("v1")
                self.assertEqual(len(list_variants_for_base("svc.base.node")), 1)

                import_from_json(str(out), mode="replace")
                self.assertEqual(len(list_variants_for_base("svc.base.node")), 2)


if __name__ == "__main__":
    unittest.main()
