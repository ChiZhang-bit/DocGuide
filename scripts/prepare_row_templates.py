"""Generate row-description templates once for precisely the selected schemas."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from utils.common import HOME, require, write_json
from utils.data.loader import load_dataset, select_indices
from utils.data.sources import Sources
from utils.llm.client import LoggedLLM
from utils.llm.config import get_model_name, get_request_timeout
from .decomposition_retrieval.row_templates import RowTemplateStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, help="Inclusive; defaults to the last dataset record.")
    parser.add_argument("--model", default=get_model_name())
    parser.add_argument("--api-mode", choices=("chat", "responses"),
                        default=os.getenv("DOCQA_LLM_API_MODE", "chat"))
    parser.add_argument("--timeout", type=float, default=get_request_timeout())
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.getenv("DOCQA_ROW_TEMPLATE_CACHE_DIR") or HOME/".cache"/"row_templates"))
    parser.add_argument("--output", type=Path, help="Optional preprocessing summary inside the repository.")
    parser.add_argument("--retry-fallback", action="store_true",
                        help="Explicitly retry cached fallback schemas; ordinary runs always reuse them.")
    args = parser.parse_args(argv)
    args.timeout = get_request_timeout(args.timeout)
    require(args.model, "Set model in utils/llm/config.py or pass --model")
    rows = load_dataset(args.data)
    indices = select_indices(rows, args.indices, args.start_index, args.end_index)
    stores = {}
    for index in indices:
        source = Sources(rows[index])
        store = RowTemplateStore(source.headers, args.model, args.cache_dir)
        stores.setdefault(store.key, store)
    llm = None
    results = []
    for store in stores.values():
        if args.retry_fallback:
            store.retry_fallback()
        try:
            diagnostics = store.require_prepared()
        except ValueError:
            if llm is None:
                llm = LoggedLLM(args.model, timeout=args.timeout, api_mode=args.api_mode)
            diagnostics = store.prepare(llm)
        results.append(diagnostics)
    result = {"data": str(args.data.resolve()), "selected_indices": indices,
              "selected_questions": len(indices), "unique_schemas": len(stores),
              "cache_dir": str(args.cache_dir.resolve()), "schemas": results}
    if args.output:
        write_json(args.output, result)
    print(json.dumps({"selected_questions": len(indices), "unique_schemas": len(stores),
                      "cached_fallback_schemas": sum(item["fallback"] for item in results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
