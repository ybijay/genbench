Complete both parts concisely:

PART 1 (Algorithmic Analysis): Explain the difference in time complexity, space complexity, and worst-case scenarios between QuickSort and MergeSort, detailing how pivot selection (first/last, random, median-of-three) affects QuickSort.

PART 2 (Agentic Diagnostic Dispatch): A production container `checkout-service` on node `prod-us-east-04` is returning HTTP 502 errors and high memory alerts. Output a fenced ```json block containing a JSON array of tool call objects to invoke in parallel from these available tools:
- `get_container_logs` (args: `service_name` [str], `node_id` [str], `tail_lines` [int])
- `get_node_metrics` (args: `node_id` [str], `metric_types` [list of 'cpu','memory','disk','network'])
- `get_deployment_history` (args: `service_name` [str], `limit` [int])
Each array item must follow: {"name": "<tool_name>", "arguments": {...}}.
