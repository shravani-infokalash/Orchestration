import json
import time
import logging
import random
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("databricks_formula_executor")

# CREATE WIDGET FOR PAYLOAD
dbutils.widgets.text("payload", "{}", "JSON Payload")


def parse_sql_query(query: str) -> Dict[str, Any]:
    """Parse SQL query to understand structure and generate appropriate dummy data"""
    query_lower = query.lower().strip()
    
    # Extract column names from SELECT statement
    columns = []
    if 'select' in query_lower:
        # Simple regex to extract columns (basic implementation)
        select_part = re.search(r'select\s+(.*?)\s+from', query_lower, re.DOTALL)
        if select_part:
            columns_str = select_part.group(1)
            # Split by comma and clean up
            raw_columns = [col.strip() for col in columns_str.split(',')]
            for col in raw_columns:
                if col == '*':
                    # If SELECT *, generate some common columns
                    columns = ['id', 'product_id', 'value', 'amount', 'quantity', 'date_created']
                    break
                else:
                    # Extract column name (remove aliases, functions, etc.)
                    clean_col = re.sub(r'\s+as\s+\w+', '', col)
                    clean_col = re.sub(r'[^\w.]', '_', clean_col)
                    columns.append(clean_col[:50])  # Limit column name length
    
    # If no columns found, use defaults
    if not columns:
        columns = ['id', 'result', 'value']
    
    # Determine number of rows based on query complexity
    if any(keyword in query_lower for keyword in ['count', 'sum', 'avg', 'group by']):
        row_count = random.randint(1, 10)  # Aggregation queries return fewer rows
    elif 'limit' in query_lower:
        # Try to extract limit number
        limit_match = re.search(r'limit\s+(\d+)', query_lower)
        if limit_match:
            row_count = min(int(limit_match.group(1)), 1000)
        else:
            row_count = random.randint(10, 100)
    else:
        row_count = random.randint(50, 500)
    
    return {
        'columns': columns,
        'row_count': row_count
    }


def generate_dummy_value(column_name: str, row_index: int) -> Any:
    """Generate appropriate dummy value based on column name patterns"""
    column_lower = column_name.lower()
    
    # ID columns
    if 'id' in column_lower:
        return random.randint(1000, 99999)
    
    # Date columns
    if any(date_keyword in column_lower for date_keyword in ['date', 'time', 'created', 'updated']):
        base_date = datetime.now()
        days_offset = random.randint(-365, 0)
        result_date = datetime(base_date.year, base_date.month, base_date.day) 
        return (result_date.replace(day=1) if days_offset < -30 else result_date).isoformat()
    
    # Amount/Money columns
    if any(money_keyword in column_lower for money_keyword in ['amount', 'price', 'cost', 'revenue', 'profit']):
        return round(random.uniform(10.0, 10000.0), 2)
    
    # Quantity/Count columns
    if any(qty_keyword in column_lower for qty_keyword in ['quantity', 'count', 'number', 'qty']):
        return random.randint(1, 100)
    
    # Percentage columns
    if any(pct_keyword in column_lower for pct_keyword in ['percent', 'rate', 'ratio']):
        return round(random.uniform(0.0, 100.0), 2)
    
    # Status/Category columns
    if any(status_keyword in column_lower for status_keyword in ['status', 'state', 'category', 'type']):
        statuses = ['active', 'inactive', 'pending', 'completed', 'processing']
        return random.choice(statuses)
    
    # Name columns
    if 'name' in column_lower:
        names = ['Product A', 'Product B', 'Service X', 'Item Y', 'Component Z']
        return f"{random.choice(names)} {row_index}"
    
    # Boolean columns
    if any(bool_keyword in column_lower for bool_keyword in ['is_', 'has_', 'active', 'enabled']):
        return random.choice([True, False])
    
    # Default: numeric value
    return round(random.uniform(1.0, 1000.0), 2)


def execute_sql_comprehensive(query: str, debug: bool = False, client_id: str = "", product_id: str = None, task_key: str = None) -> Dict[str, Any]:
    """Generate dummy SQL results instead of executing against SQL warehouse"""
    try:
        start_time = time.time()
        
        logger.info(f"Generating dummy data for query at {datetime.now().strftime('%H:%M:%S')} - Task: {task_key}, Product: {product_id}")
        
        if debug:
            logger.debug(f"Query: {query}")
        
        # Add random delay to simulate execution time
        execution_delay = random.uniform(0.5, 3.0)
        time.sleep(execution_delay)
        
        # Parse query to determine structure
        query_info = parse_sql_query(query)
        columns = query_info['columns']
        row_count = query_info['row_count']
        
        # Generate dummy data
        typed_rows = []
        for i in range(row_count):
            row = {}
            for col in columns:
                row[col] = generate_dummy_value(col, i + 1)
            typed_rows.append(row)
        
        # Create column type information
        column_types = []
        for col in columns:
            col_lower = col.lower()
            if 'id' in col_lower:
                column_types.append('BIGINT')
            elif any(date_keyword in col_lower for date_keyword in ['date', 'time']):
                column_types.append('TIMESTAMP')
            elif any(bool_keyword in col_lower for bool_keyword in ['is_', 'has_', 'active', 'enabled']):
                column_types.append('BOOLEAN')
            else:
                column_types.append('DOUBLE')
        
        execution_time = time.time() - start_time
        
        inline_data = {
            "rows": typed_rows,
            "row_count": len(typed_rows),
            "columns": columns,
            "format": "JSON_ARRAY"
        }
        
        # Generate dummy external links
        external_links = [
            f"https://test-storage.example.com/results/{task_key}_{product_id}_{int(time.time())}.csv"
        ]
        
        # Generate dummy manifest
        manifest = {
            "format": "CSV",
            "total_row_count": row_count,
            "total_chunk_count": 1,
            "chunks": [{
                "chunk_index": 0,
                "row_offset": 0,
                "row_count": row_count
            }],
            "schema": {
                "columns": [
                    {"name": col, "type_text": col_type, "type_name": col_type}
                    for col, col_type in zip(columns, column_types)
                ]
            }
        }
        
        return {
            "status": "SUCCEEDED",
            "execution_time_seconds": execution_time,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "inline": inline_data,
            "external_links": external_links,
            "manifest": manifest,
            "format": "BOTH",
            "row_count": row_count,
            "error": None,
            "product_id": product_id,
            "task_key": task_key,
            "dummy_data": True  # Flag to indicate this is dummy data
        }

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error generating dummy data for task {task_key}, product {product_id}: {str(e)}")

        return {
            "status": "FAILED",
            "execution_time_seconds": execution_time,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": None,
            "error": str(e),
            "product_id": product_id,
            "task_key": task_key,
            "dummy_data": True
        }


def process_task(task: Dict[str, Any], workflow_run_id: str, debug: bool = False) -> Dict[str, Any]:
    """Process an individual task from the workflow"""
    task_key = task.get("task_key", "")
    description = task.get("description", "")
    sql_task = task.get("sql_task", {})
    timeout_seconds = task.get("timeout_seconds", 30)
    
    # Extract product information from SQL task parameters
    sql_parameters = sql_task.get("parameters", {})
    product_id = sql_parameters.get("product_id", "")
    calculation_order = sql_parameters.get("calculation_order", "0")
    
    # Extract SQL query
    query_info = sql_task.get("query", {})
    sql_query = query_info.get("query", "")
    
    logger.info(f"Processing task: {task_key} - {description}")
    logger.info(f"Product ID: {product_id}, Calculation Order: {calculation_order}")
    
    task_result = {
        "task_key": task_key,
        "product_id": product_id,
        "description": description,
        "calculation_order": int(calculation_order) if calculation_order.isdigit() else 0,
        "timeout_seconds": timeout_seconds,
        "status": "PENDING",
        "dependencies": [dep.get("task_key") for dep in task.get("depends_on", [])],
        "sql_result": None,
        "error": None,
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }
    
    # Execute the SQL query if it exists (now with dummy data)
    if sql_query:
        try:
            # Determine client_id from workflow tags or default
            client_id = workflow_run_id.split('_')[-1]  # Extract from workflow_run_id
            
            sql_result = execute_sql_comprehensive(
                query=sql_query,
                debug=debug,
                client_id=client_id,
                product_id=product_id,
                task_key=task_key
            )
            
            task_result["sql_result"] = sql_result
            task_result["status"] = sql_result.get("status", "UNKNOWN")
            
            if sql_result.get("status") == "FAILED":
                task_result["error"] = sql_result.get("error")
                
        except Exception as e:
            logger.error(f"Error processing task {task_key}: {str(e)}")
            task_result["status"] = "FAILED"
            task_result["error"] = str(e)
    else:
        # If no SQL query, mark as completed
        task_result["status"] = "SUCCEEDED"
    
    task_result["end_time"] = datetime.now().isoformat()
    return task_result


def webhook_notify_individual_task(task_result: Dict[str, Any], workflow_info: Dict[str, Any], workflow_payload: Dict[str, Any]):
    """Send individual task result to webhook"""
    webhook_url = "https://dev.morpheo.ai/api/webhook/job-status"  # Replace with actual webhook URL

    task_key = task_result.get("task_key")
    product_id = task_result.get("product_id")
    sql_result = task_result.get("sql_result", {})
    
    # Extract workflow information
    workflow_run_id = workflow_info.get("workflow_run_id", "")
    client_id = workflow_info.get("client_id", "")
    
    # Get email based on task status
    email_notifications = workflow_payload.get("email_notifications", {})
    task_status = sql_result.get("status", "UNKNOWN") if sql_result else "NO_QUERY"
    
    # Select appropriate email based on status
    if task_status in ["FAILED", "ERROR"]:
        emails = email_notifications.get("on_failure", [""])
        logger.info(f"Task {task_key} failed, using failure emails: {emails}")
    elif task_status == "SUCCEEDED":
        emails = email_notifications.get("on_success", [""])
        logger.info(f"Task {task_key} succeeded, using success emails: {emails}")
    else:
        # For unknown or pending status, use success email as default
        emails = email_notifications.get("on_success", [""])
        logger.info(f"Task {task_key} has status {task_status}, using success emails as default: {emails}")
    
    client_email = emails[0] if emails else ""
    logger.info(f"Selected email for task {task_key}: {client_email}")
    
    # Prepare individual task payload
    individual_payload = {
        "product_id": product_id,
        "EstimatedTime": sql_result.get("execution_time_seconds", 0),
        "client_id": client_id,
        "created_at": datetime.now().isoformat(),
        "email": client_email,
        "name": task_result.get("description", ""),
        "result": {
            "execution_summary": {
                "end_time": task_result.get("end_time"),
                "failed_queries": 1 if sql_result.get("status") == "FAILED" else 0,
                "start_time": task_result.get("start_time"),
                "successful_queries": 1 if sql_result.get("status") == "SUCCEEDED" else 0,
                "total_execution_time": sql_result.get("execution_time_seconds", 0),
                "total_queries": 1
            },
            "task": {
                "task_key": task_key,
                "product_id": product_id,
                "description": task_result.get("description", ""),
                "calculation_order": task_result.get("calculation_order", 0),
                "timeout_seconds": task_result.get("timeout_seconds", 30),
                "dependencies": task_result.get("dependencies", []),
                "sql_result": sql_result,  # Include the complete SQL result data
                "error": sql_result.get("error") if sql_result else None
            },
        },
        # Add SQL results directly to the payload root for easier access
        "data": sql_result.get("inline", {}).get("rows", []) if sql_result else [],
        "row_count": sql_result.get("row_count", 0) if sql_result else 0,
        "columns": sql_result.get("inline", {}).get("columns", []) if sql_result else [],
        "external_links": sql_result.get("external_links", []) if sql_result else [],
        "query": sql_result.get("query", "") if sql_result else "",
        "sql_execution_time": sql_result.get("execution_time_seconds", 0) if sql_result else 0,
        "status": sql_result.get("status", "UNKNOWN") if sql_result else "NO_QUERY",
        "run_id": "",
        "status": "completed" if sql_result.get("status") == "SUCCEEDED" else "failed",
        "workflow_run_id": workflow_run_id,
        "task_key": task_key
    }

    try:
        logger.info(f"Attempting to send webhook for task_key={task_key}, product_id={product_id}, email={client_email}")
        logger.info(f"Webhook URL: {webhook_url}")
        logger.debug(f"Webhook payload: {json.dumps(individual_payload, indent=2)}")
        
        # Make actual HTTP request to webhook
        import requests
        response = requests.post(webhook_url, json=individual_payload, timeout=30)
        response.raise_for_status()
        
        logger.info(f"Individual webhook sent successfully for {task_key}. Status: {response.status_code}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error sending individual webhook for {task_key}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending individual webhook for {task_key}: {e}")
        return False


def build_dependency_graph(tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Build a dependency graph from tasks"""
    dependency_graph = {}
    
    for task in tasks:
        task_key = task.get("task_key")
        dependencies = [dep.get("task_key") for dep in task.get("depends_on", [])]
        dependency_graph[task_key] = dependencies
    
    return dependency_graph


def topological_sort(dependency_graph: Dict[str, List[str]]) -> List[str]:
    """Perform topological sort to determine execution order"""
    from collections import deque, defaultdict
    
    # Get all tasks from the dependency graph
    all_tasks = set(dependency_graph.keys())
    
    # Calculate in-degrees (how many dependencies each task has)
    in_degree = defaultdict(int)
    
    # Initialize all tasks with 0 in-degree
    for task in all_tasks:
        in_degree[task] = 0
    
    # Count dependencies for each task
    for task in dependency_graph:
        dependencies = dependency_graph[task]
        in_degree[task] = len(dependencies)
        
        # Ensure all dependency tasks are in our task set
        for dep in dependencies:
            if dep not in all_tasks:
                logger.warning(f"Dependency {dep} for task {task} not found in task list")
    
    # Find tasks with no dependencies (in-degree = 0)
    queue = deque([task for task in all_tasks if in_degree[task] == 0])
    execution_order = []
    
    logger.info(f"Initial queue (no dependencies): {list(queue)}")
    logger.info(f"In-degrees: {dict(in_degree)}")
    
    while queue:
        current_task = queue.popleft()
        execution_order.append(current_task)
        logger.debug(f"Processing task: {current_task}")
        
        # Find tasks that depend on the current task and reduce their in-degree
        for task in dependency_graph:
            if current_task in dependency_graph[task]:
                in_degree[task] -= 1
                logger.debug(f"Reduced in-degree for {task} to {in_degree[task]}")
                if in_degree[task] == 0:
                    queue.append(task)
                    logger.debug(f"Added {task} to queue")
    
    # Check for circular dependencies
    if len(execution_order) != len(all_tasks):
        remaining_tasks = all_tasks - set(execution_order)
        logger.error(f"Circular dependency detected! Remaining tasks: {remaining_tasks}")
        logger.error(f"In-degrees of remaining tasks: {[(task, in_degree[task]) for task in remaining_tasks]}")
        # Add remaining tasks to execution order anyway (they'll fail with dependency errors)
        execution_order.extend(remaining_tasks)
    
    return execution_order


def process_workflow(workflow_payload: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """Process the entire workflow with proper dependency resolution"""
    start_time = time.time()
    
    # Extract workflow information
    workflow_name = workflow_payload.get("name", "")
    workflow_tags = workflow_payload.get("tags", {})
    workflow_run_id = workflow_tags.get("workflow_run_id", f"wf_run_{int(time.time())}")
    main_product_id = workflow_tags.get("main_product_id", "")
    
    # Extract client info from email notifications or tags
    email_notifications = workflow_payload.get("email_notifications", {})
    emails = email_notifications.get("on_success", [""])
    client_email = emails[0] if emails else ""
    client_id = workflow_run_id.split('_')[-1]  # Extract from workflow_run_id
    
    tasks = workflow_payload.get("tasks", [])
    
    logger.info(f"Processing workflow: {workflow_name}")
    logger.info(f"Workflow Run ID: {workflow_run_id}")
    logger.info(f"Main Product ID: {main_product_id}")
    logger.info(f"Total Tasks: {len(tasks)}")
    logger.info("NOTE: Using dummy data generation instead of SQL warehouse")
    
    workflow_info = {
        "workflow_run_id": workflow_run_id,
        "client_id": client_id,
        "email": client_email,
        "main_product_id": main_product_id
    }
    
    # Build dependency graph and determine execution order
    dependency_graph = build_dependency_graph(tasks)
    execution_order = topological_sort(dependency_graph)
    
    logger.info(f"Execution order determined: {execution_order}")
    
    # Create task lookup
    task_lookup = {task.get("task_key"): task for task in tasks}
    
    # Initialize result structure
    workflow_result = {
        "workflow_name": workflow_name,
        "workflow_run_id": workflow_run_id,
        "main_product_id": main_product_id,
        "client_id": client_id,
        "email": client_email,
        "execution_summary": {
            "total_tasks": len(tasks),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_execution_time": 0,
            "webhooks_sent": []
        },
        "task_results": [],
        "execution_order": execution_order
    }
    
    # Process tasks in dependency order
    completed_tasks = set()
    failed_tasks = set()
    
    try:
        for task_key in execution_order:
            if task_key not in task_lookup:
                logger.warning(f"Task {task_key} not found in task lookup, skipping")
                continue
                
            task = task_lookup[task_key]
            
            # Check if all dependencies are completed successfully
            dependencies = dependency_graph.get(task_key, [])
            failed_dependencies = []
            
            for dep_task_key in dependencies:
                if dep_task_key in failed_tasks:
                    failed_dependencies.append(dep_task_key)
                elif dep_task_key not in completed_tasks:
                    # This shouldn't happen with proper topological sort, but handle it
                    logger.warning(f"Dependency {dep_task_key} for task {task_key} not yet completed")
                    failed_dependencies.append(dep_task_key)
            
            if failed_dependencies:
                logger.error(f"Task {task_key} cannot execute - failed dependencies: {failed_dependencies}")
                task_result = {
                    "task_key": task_key,
                    "product_id": task.get("sql_task", {}).get("parameters", {}).get("product_id", ""),
                    "description": task.get("description", ""),
                    "status": "FAILED",
                    "error": f"Dependencies failed: {failed_dependencies}",
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "dependencies": dependencies,
                    "calculation_order": int(task.get("sql_task", {}).get("parameters", {}).get("calculation_order", "0")),
                    "timeout_seconds": task.get("timeout_seconds", 30),
                    "sql_result": {"status": "FAILED", "error": f"Dependencies failed: {failed_dependencies}"}
                }
                failed_tasks.add(task_key)
                workflow_result["execution_summary"]["failed_tasks"] += 1
            else:
                # Process the task
                logger.info(f"Executing task {task_key} - all dependencies satisfied: {dependencies}")
                task_result = process_task(task, workflow_run_id, debug)
                
                if task_result.get("status") == "SUCCEEDED":
                    completed_tasks.add(task_key)
                    workflow_result["execution_summary"]["successful_tasks"] += 1
                    logger.info(f"Task {task_key} completed successfully with dummy data")
                else:
                    failed_tasks.add(task_key)
                    workflow_result["execution_summary"]["failed_tasks"] += 1
                    logger.error(f"Task {task_key} failed: {task_result.get('error', 'Unknown error')}")
            
            workflow_result["task_results"].append(task_result)
            
            # Send individual webhook for this task with workflow_payload
            logger.info(f"About to send webhook for task {task_key}")
            webhook_sent = webhook_notify_individual_task(task_result, workflow_info, workflow_payload)
            workflow_result["execution_summary"]["webhooks_sent"].append({
                "task_key": task_key,
                "product_id": task_result.get("product_id"),
                "webhook_sent": webhook_sent,
                "status": task_result.get("status")
            })
            logger.info(f"Webhook result for task {task_key}: {'sent' if webhook_sent else 'failed'}")
    
    except Exception as e:
        logger.error(f"Error processing workflow: {str(e)}")
        workflow_result["error"] = str(e)
    
    # Finalize execution summary
    total_time = time.time() - start_time
    workflow_result["execution_summary"]["end_time"] = datetime.now().isoformat()
    workflow_result["execution_summary"]["total_execution_time"] = total_time
    
    # Determine overall status
    if workflow_result["execution_summary"]["failed_tasks"] == 0:
        overall_status = "SUCCESS"
    elif workflow_result["execution_summary"]["successful_tasks"] > 0:
        overall_status = "PARTIAL_SUCCESS"
    else:
        overall_status = "FAILED"
    
    workflow_result["overall_status"] = overall_status
    
    logger.info(f"Workflow processing completed in {total_time:.2f}s with dummy data")
    logger.info(f"Summary: {workflow_result['execution_summary']['successful_tasks']}/{workflow_result['execution_summary']['total_tasks']} tasks successful")
    
    # Return summary for notebook output
    successful_webhooks = sum(1 for w in workflow_result["execution_summary"]["webhooks_sent"] if w["webhook_sent"])
    
    return {
        "processing_summary": {
            "workflow_run_id": workflow_run_id,
            "workflow_name": workflow_name,
            "total_tasks_processed": len(workflow_result["task_results"]),
            "successful_tasks": workflow_result["execution_summary"]["successful_tasks"],
            "failed_tasks": workflow_result["execution_summary"]["failed_tasks"],
            "successful_webhooks": successful_webhooks,
            "failed_webhooks": len(workflow_result["execution_summary"]["webhooks_sent"]) - successful_webhooks,
            "execution_time": total_time,
            "status": overall_status,
            "dummy_data_mode": True
        },
        "individual_webhooks_sent": workflow_result["execution_summary"]["webhooks_sent"],
        "execution_order": execution_order
    }


# Databricks notebook entry point
try:
    payload_str = dbutils.widgets.get("payload")
    logger.info(f"Raw payload string length: {len(payload_str) if payload_str else 0}")
    logger.info(f"Raw payload preview (first 200 chars): {payload_str[:200] if payload_str else 'None'}")
    
    # Enhanced JSON parsing with better error handling
    if not payload_str or payload_str.strip() in ["{}", ""]:
        raise ValueError("Empty or default payload received")
    
    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError as je:
        logger.error(f"JSON decode error at position {je.pos}: {je.msg}")
        logger.error(f"Problem area: '{payload_str[max(0, je.pos-50):je.pos+50]}'")
        
        # Try to fix common JSON issues
        cleaned_payload = payload_str.strip()
        
        # Remove any trailing commas before closing brackets/braces
        import re
        cleaned_payload = re.sub(r',(\s*[}\]])', r'\1', cleaned_payload)
        
        # Try parsing the cleaned version
        try:
            payload = json.loads(cleaned_payload)
            logger.info("Successfully parsed JSON after cleaning")
        except json.JSONDecodeError as je2:
            logger.error(f"Still failed after cleaning at position {je2.pos}: {je2.msg}")
            logger.error(f"Full payload causing error: {payload_str}")
            raise ValueError(f"Invalid JSON payload: {je2.msg} at position {je2.pos}")

    # Check if this is the new workflow format
    if "tasks" in payload and "name" in payload:
        logger.info("Detected new workflow format payload - Running in DUMMY DATA mode")
        debug = payload.get("debug", False)
        
        result = process_workflow(
            workflow_payload=payload,
            debug=debug
        )
        
        # The result contains a processing summary for the workflow
        final_result = {
            "status": "SUCCESS" if result["processing_summary"]["status"] not in ["FAILED", "ERROR"] else "ERROR",
            "message": f"Processed workflow with {result['processing_summary']['total_tasks_processed']} tasks using DUMMY DATA",
            "processing_summary": result["processing_summary"],
            "individual_webhooks": result["individual_webhooks_sent"],
            "execution_order": result["execution_order"],
            "timestamp": datetime.now().isoformat(),
            "dummy_data_mode": True
        }
        
    else:
        # Legacy format - keep existing formula processing logic
        logger.info("Detected legacy formula format payload")
        raise ValueError("Legacy formula format is no longer supported. Please use the new workflow format.")

    print("=== WORKFLOW PROCESSING RESULT (DUMMY DATA MODE) ===")
    print(json.dumps(final_result, indent=2, default=str))

    # Simplified result for notebook exit
    simple_result = {
        "status": final_result["status"],
        "workflow_run_id": result["processing_summary"]["workflow_run_id"],
        "workflow_name": result["processing_summary"]["workflow_name"],
        "total_tasks_processed": result["processing_summary"]["total_tasks_processed"],
        "successful_tasks": result["processing_summary"]["successful_tasks"],
        "failed_tasks": result["processing_summary"]["failed_tasks"],
        "successful_webhooks": result["processing_summary"]["successful_webhooks"],
        "failed_webhooks": result["processing_summary"]["failed_webhooks"],
        "execution_time": result["processing_summary"]["execution_time"],
        "message": "Workflow tasks processed with dummy data and individual webhooks sent",
        "dummy_data_mode": True
    }

    # dbutils.notebook.exit(json.dumps(simple_result, default=str))

except Exception as e:
    logger.error(f"Error processing payload: {str(e)}")
    error_result = {
        "status": "FAILED",
        "error": str(e),
        "timestamp": datetime.now().isoformat(),
        "dummy_data_mode": True
    }

    print("=== ERROR RESULT ===")
    print(json.dumps(error_result, indent=2))
    
    # dbutils.notebook.exit(json.dumps(error_result))