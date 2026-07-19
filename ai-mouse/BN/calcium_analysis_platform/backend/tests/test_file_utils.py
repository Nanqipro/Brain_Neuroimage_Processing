from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.file_utils import (
    InvalidFile,
    copy_limited,
    resolve_existing_file,
    safe_client_filename,
    unique_upload_path,
)


class FileUtilsTests(unittest.TestCase):
    def test_client_path_is_reduced_to_basename(self):
        self.assertEqual(safe_client_filename("../../data/sample.xlsx"), "sample.xlsx")
        self.assertEqual(safe_client_filename(r"..\data\sample.xls"), "sample.xls")

    def test_unsupported_suffix_is_rejected(self):
        with self.assertRaises(InvalidFile):
            safe_client_filename("payload.py")

    def test_unique_upload_path_stays_below_storage_root(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = unique_upload_path(root, "../../sample.xlsx")
            self.assertEqual(destination.parent, root)
            self.assertTrue(destination.name.endswith("_sample.xlsx"))

    def test_resolve_existing_file_rejects_missing_or_wrong_type(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = root / "result.xlsx"
            expected.write_bytes(b"test")
            self.assertEqual(resolve_existing_file(root, "result.xlsx"), expected.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_existing_file(root, "missing.xlsx")
            with self.assertRaises(InvalidFile):
                resolve_existing_file(root, "result.txt")

    def test_copy_limited_removes_partial_file(self):
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "large.xlsx"
            with self.assertRaises(InvalidFile):
                copy_limited(BytesIO(b"12345"), destination, max_bytes=4, chunk_size=2)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
