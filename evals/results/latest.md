# Evaluation results

Run at 2026-10-23T17:58:10 · answers `gemini-3.5-flash` · judge `gemini-3.5-flash-lite` · embeddings `gemini-embedding-001` · top_k 3 · chunk size 256 · 12/12 questions

| Metric | Value |
|---|---|
| Retrieval hit rate (%) | 90.0 |
| Mean reciprocal rank | 0.85 |
| Faithfulness (%) | 70.0 |
| Relevancy (%) | 100 |
| Correctness (1 to 5) | 3.9 |
| Out-of-scope refusal (%) | 100 |

## Per question

| ID | Hit | Faithful | Relevant | Correctness | Refused | Judge note |
|---|---|---|---|---|---|---|
| q01 | yes | yes | yes | 5 |  | The chatbot accurately and completely answers the question using the provided context. |
| q02 | yes | yes | yes | 5 |  | The answer accurately provides the exact location of internship openings found in the context. |
| q03 | yes | yes | yes | 5 |  | The chatbot's answer perfectly matches the reference answer and is fully supported by the context. |
| q04 | no | yes | yes | 1 |  | The chatbot failed to answer the question using the available context and incorrectly stated the information was missing. |
| q05 | yes | no | yes | 4 |  | The answer includes processing time details not found in the provided context. |
| q06 | yes | no | yes | 1 |  | The retrieved context does not contain the processing time, making the answer unfaithful to the provided context. |
| q07 | yes | no | yes | 3 |  | The answer is correct and relevant, but the claim about timing is not supported by the provided context. |
| q08 | yes | yes | yes | 5 |  | The chatbot accurately and completely answers the question using only facts from the retrieved context. |
| q09 | yes | yes | yes | 5 |  | The chatbot accurately and completely answers who handles and approves the documents based on the text. |
| q10 | yes | yes | yes | 5 |  | The answer is direct, accurate, and fully supported by the context. |
| q11 |  |  |  |  | yes |  |
| q12 |  |  |  |  | yes |  |
