import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools import ai_artifact_assistant as assistant


class ArtifactAssistantTests(unittest.TestCase):
    def test_chunk_text_uses_overlap(self):
        text = "0123456789" * 20

        chunks = assistant.chunk_text(text, max_chars=50, overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][-10:], chunks[1][:10])

    def test_lexical_score_matches_mes_terms(self):
        score = assistant.lexical_score(
            "MES integration SAP",
            "The MES project integrates production orders with SAP ERP.",
        )

        self.assertGreater(score, 0)

    def test_build_index_without_embeddings_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "handover.md"
            db_path = root / "index.json"
            artifact.write_text(
                "# Handover\nMES receives production orders from SAP and returns confirmations.",
                encoding="utf-8",
            )
            args = Namespace(
                sources=[str(root)],
                db=str(db_path),
                extensions=None,
                chunk_chars=500,
                overlap=50,
                max_file_bytes=2_000_000,
                no_embeddings=True,
                ollama_url=assistant.DEFAULT_OLLAMA_URL,
                timeout=1,
                embed_model=assistant.DEFAULT_EMBED_MODEL,
            )

            result = assistant.build_index(args)
            index = json.loads(db_path.read_text(encoding="utf-8"))
            ranked = assistant.rank_chunks(index, "SAP confirmations", top_k=3)

            self.assertEqual(result, 0)
            self.assertEqual(len(index["files"]), 1)
            self.assertIsNone(index["embedding_model"])
            self.assertEqual(ranked[0][1]["source"], os.path.relpath(artifact, Path.cwd()))


if __name__ == "__main__":
    unittest.main()
