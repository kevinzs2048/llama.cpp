import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from export_flaggems_aot import unique_object, validate_profile

class ExportProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"export":"example", "semantic":"example.v1", "signature":{"source":"*fp32","count":"i32"}, "constants":{"BLOCK":256}, "geometry":{"block_size":"BLOCK"}, "num_warps":4}
        self.export = SimpleNamespace(name="example",semantic="example.v1",runtime_arguments=("source","count"),constexpr_arguments=("BLOCK",),runtime_types=("*fp32","i32"))
    def test_geometry_comes_from_compiled_constants(self):
        name,signature,constants,geometry,warps=validate_profile(self.profile,self.export)
        self.assertEqual((name,geometry,warps),("example",{"block_size":256},4))
    def test_runtime_types_match_library_descriptor(self):
        p=copy.deepcopy(self.profile);p["signature"]["source"]="*fp16"
        with self.assertRaisesRegex(RuntimeError,"runtime type mismatch"):validate_profile(p,self.export)
    def test_unknown_argument_cannot_change_the_abi(self):
        p=copy.deepcopy(self.profile);p["signature"]["other"]="i32"
        with self.assertRaises(RuntimeError):validate_profile(p,self.export)
    def test_changed_semantic_is_rejected(self):
        p=copy.deepcopy(self.profile);p["semantic"]="different.v1"
        with self.assertRaises(RuntimeError):validate_profile(p,self.export)
    def test_missing_constant_is_rejected(self):
        p=copy.deepcopy(self.profile);p["constants"]={}
        with self.assertRaises(RuntimeError):validate_profile(p,self.export)
    def test_unknown_geometry_reference_is_rejected(self):
        p=copy.deepcopy(self.profile);p["geometry"]["block_size"]="other"
        with self.assertRaises(RuntimeError):validate_profile(p,self.export)
    def test_negative_geometry_is_rejected(self):
        p=copy.deepcopy(self.profile);p["constants"]["BLOCK"]=-1
        with self.assertRaises(RuntimeError):validate_profile(p,self.export)
    def test_boolean_warp_count_is_rejected(self):
        p=copy.deepcopy(self.profile);p["num_warps"]=True
        with self.assertRaises(RuntimeError):validate_profile(p,self.export)
    def test_duplicate_json_field_is_rejected(self):
        with self.assertRaises(RuntimeError):json.loads('{"a":1,"a":2}',object_pairs_hook=unique_object)
    def test_legacy_untyped_descriptor_still_works(self):
        self.export.runtime_types=()
        self.assertEqual(validate_profile(self.profile,self.export)[0],"example")

if __name__=="__main__":unittest.main()
