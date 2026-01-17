#!/usr/bin/env python3
"""CLI for Web Intelligence Suite."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console

from core.logging import setup_logging

from .core import Web, WebConfig
from .workflows import (
    WorkflowEngine,
    WorkflowDefinition,
    create_unreal_remote_control_workflow,
)
from .autolearn import AutoLearner

app = typer.Typer(
    name="web",
    help="Web Intelligence Suite - Standalone web research tool",
    add_completion=False,
)
console = Console()

# Workflow sub-app
workflow_app = typer.Typer(name="workflow", help="Workflow operations")
app.add_typer(workflow_app)


def parse_json_arg(value: str) -> dict:
    """Parse JSON string argument."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Try as key=value pairs
        result = {}
        for pair in value.split(","):
            if "=" in pair:
                key, val = pair.split("=", 1)
                result[key.strip()] = val.strip()
        return result


async def run_search(web: Web, query: str) -> None:
    """Run search operation."""
    try:
        if not query or not query.strip():
            logger.error("Search query cannot be empty")
            raise ValueError("Search query cannot be empty")
        
        logger.info(f"Starting search for: {query}")
        result = await web.execute("search", query=query)
        logger.info("Search completed. Found results.")
        console.print("[green]✓ Search completed successfully[/green]")
        console.print(result)
    except ValueError as e:
        logger.error(f"Invalid search query: {e}")
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        console.print(f"[red]✗ Search failed: {e}[/red]")
        raise typer.Exit(1)


async def run_scrape(web: Web, url: str) -> None:
    """Run scrape operation."""
    try:
        if not url or not url.strip():
            logger.error("URL cannot be empty")
            raise ValueError("URL cannot be empty")
        
        if not url.startswith(("http://", "https://")):
            logger.error(f"Invalid URL format: {url}")
            raise ValueError("URL must start with http:// or https://")
        
        logger.info(f"Starting scrape of: {url}")
        result = await web.execute("scrape", url=url)
        logger.info("Scrape completed successfully")
        console.print("[green]✓ Scrape completed successfully[/green]")
        console.print(result)
    except ValueError as e:
        logger.error(f"Invalid URL: {e}")
        console.print(f"[red]✗ Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Scrape failed: {e}")
        console.print(f"[red]✗ Scrape failed: {e}[/red]")
        raise typer.Exit(1)


async def run_api(
    web: Web,
    url: str,
    method: str = "GET",
    headers: Optional[str] = None,
    json_data: Optional[str] = None,
    params: Optional[str] = None,
    auth: Optional[str] = None,
) -> None:
    """Run API operation."""
    parsed_headers = parse_json_arg(headers) if headers else None
    parsed_json = json.loads(json_data) if json_data else None
    parsed_params = parse_json_arg(params) if params else None
    parsed_auth = parse_json_arg(auth) if auth else None

    result = await web.execute(
        "api",
        url=url,
        method=method,
        headers=parsed_headers,
        json_data=parsed_json,
        params=parsed_params,
        auth=parsed_auth,
    )
    logger.info(result)


async def run_discover(web: Web, url: str) -> None:
    """Run API discovery operation."""
    result = await web.execute("discover", url=url)
    logger.info(result)


async def run_workflow_create(
    web: Web,
    workflow_type: Optional[str] = None,
    base_url: Optional[str] = None,
    file: Optional[Path] = None,
) -> None:
    """Create a workflow."""
    engine = WorkflowEngine(web)
    
    if workflow_type == "unreal":
        base_url = base_url or "http://localhost:30010"
        workflow = create_unreal_remote_control_workflow(base_url=base_url)
        await engine.save_workflow(workflow)
        logger.success(f"Created workflow: {workflow.id}")
    elif file:
        with open(file, "r") as f:
            data = json.load(f)
            workflow = WorkflowDefinition(**data)
            await engine.save_workflow(workflow)
            logger.success(f"Created workflow from file: {workflow.id}")
    else:
        logger.error("Specify --type or --file")
        raise typer.Exit(1)


async def run_workflow_run(web: Web, workflow_id: str, vars: Optional[str] = None) -> None:
    """Run a workflow."""
    engine = WorkflowEngine(web)
    variables = json.loads(vars) if vars else None
    result = await engine.execute_workflow(workflow_id, variables)
    logger.success("Workflow executed successfully")
    logger.info(f"Execution ID: {result['execution_id']}")
    logger.info(f"Results: {json.dumps(result['results'], indent=2)}")


async def run_workflow_list(web: Web) -> None:
    """List all workflows."""
    results = web.db.execute(
        """
        SELECT id, name, api_type, success_count, failure_count, last_run
        FROM workflows ORDER BY name
        """
    ).fetchall()
    if results:
        logger.info("\nWorkflows:")
        logger.info("-" * 80)
        for row in results:
            logger.info(f"ID: {row[0]}")
            logger.info(f"  Name: {row[1]}")
            logger.info(f"  Type: {row[2]}")
            logger.info(f"  Success: {row[3]}, Failures: {row[4]}")
            logger.info(f"  Last Run: {row[5] or 'Never'}")
            logger.info("")
    else:
        logger.info("No workflows found")


async def run_workflow_show(web: Web, workflow_id: str) -> None:
    """Show workflow details."""
    engine = WorkflowEngine(web)
    workflow = await engine.load_workflow(workflow_id)
    if workflow:
        logger.info(f"\nWorkflow: {workflow.name}")
        logger.info(f"ID: {workflow.id}")
        logger.info(f"Description: {workflow.description}")
        logger.info(f"API Type: {workflow.api_type}")
        logger.info(f"Base URL: {workflow.base_url}")
        logger.info(f"\nSteps ({len(workflow.steps)}):")
        for i, step in enumerate(workflow.steps, 1):
            logger.info(f"  {i}. {step.name} ({step.type})")
            if step.on_success:
                logger.info(f"     → On success: {step.on_success}")
            if step.on_error:
                logger.info(f"     → On error: {step.on_error}")
    else:
        logger.error(f"Workflow {workflow_id} not found")
        raise typer.Exit(1)


async def run_workflow_autolearn(
    web: Web,
    base_url: str,
    workflow_id: Optional[str] = None,
    iterations: int = 5,
) -> None:
    """Run auto-learning cycle."""
    learner = AutoLearner(web)
    result = await learner.auto_learn_cycle(base_url, workflow_id, iterations)
    logger.success("\nAuto-learning cycle completed!")
    logger.info(f"Base URL: {result['base_url']}")
    logger.info(f"Iterations: {result['iterations']}")
    logger.info("\nResults summary:")
    for res in result["results"]:
        data = res.get('data', res.get('error', 'N/A'))
        logger.info(
            f"  Iteration {res['iteration']} - {res['phase']}: {data}"
        )


@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Path to DuckDB database",
    ),
    max_results: Optional[int] = typer.Option(
        None,
        "--max-results",
        help="Maximum search results",
    ),
) -> None:
    """Web Intelligence Suite - Standalone web research tool."""
    setup_logging(log_level)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Path to DuckDB database"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum search results"),
) -> None:
    """Search DuckDuckGo."""
    default_config = WebConfig()
    config = WebConfig(
        db_path=db_path or default_config.db_path,
        max_results=max_results or default_config.max_results,
    )
    web = Web(config)
    
    async def run():
        try:
            await run_search(web, query)
        finally:
            await web.close()
    
    asyncio.run(run())


@app.command()
def scrape(
    url: str = typer.Argument(..., help="URL to scrape"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Path to DuckDB database"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum search results"),
) -> None:
    """Scrape URL content."""
    default_config = WebConfig()
    config = WebConfig(
        db_path=db_path or default_config.db_path,
        max_results=max_results or default_config.max_results,
    )
    web = Web(config)
    
    async def run():
        try:
            await run_scrape(web, url)
        finally:
            await web.close()
    
    asyncio.run(run())


@app.command()
def api(
    url: str = typer.Argument(..., help="API URL"),
    method: str = typer.Option("GET", "--method", "-m", help="HTTP method"),
    headers: Optional[str] = typer.Option(None, "--headers", help="Headers as JSON or key=value pairs"),
    json_data: Optional[str] = typer.Option(None, "--json", help="JSON body for POST/PUT requests"),
    params: Optional[str] = typer.Option(None, "--params", help="Query parameters as JSON or key=value pairs"),
    auth: Optional[str] = typer.Option(None, "--auth", help="Auth config as JSON"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Path to DuckDB database"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum search results"),
) -> None:
    """Make API request."""
    default_config = WebConfig()
    config = WebConfig(
        db_path=db_path or default_config.db_path,
        max_results=max_results or default_config.max_results,
    )
    web = Web(config)
    
    async def run():
        try:
            await run_api(web, url, method, headers, json_data, params, auth)
        finally:
            await web.close()
    
    asyncio.run(run())


@app.command()
def discover(
    url: str = typer.Argument(..., help="Base URL to discover"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Path to DuckDB database"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum search results"),
) -> None:
    """Discover API endpoints."""
    default_config = WebConfig()
    config = WebConfig(
        db_path=db_path or default_config.db_path,
        max_results=max_results or default_config.max_results,
    )
    web = Web(config)
    
    async def run():
        try:
            await run_discover(web, url)
        finally:
            await web.close()
    
    asyncio.run(run())


@workflow_app.command("create")
def workflow_create(
    type: Optional[str] = typer.Option(None, "--type", help="Workflow type (e.g., unreal)"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL for API"),
    file: Optional[Path] = typer.Option(None, "--file", help="Load workflow from JSON file"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Path to DuckDB database"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum search results"),
) -> None:
    """Create a workflow."""
    default_config = WebConfig()
    config = WebConfig(
        db_path=db_path or default_config.db_path,
        max_results=max_results or default_config.max_results,
    )
    web = Web(config)
    
    async def run():
        try:
            await run_workflow_create(web, type, base_url, file)
        finally:
            await web.close()
    
    asyncio.run(run())


@workflow_app.command("run")
def workflow_run(
    workflow_id: str = typer.Argument(..., help="Workflow ID to execute"),
    vars: Optional[str] = typer.Option(None, "--vars", help="Variables as JSON"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Path to DuckDB database"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum search results"),
) -> None:
    """Run a workflow."""
    default_config = WebConfig()
    config = WebConfig(
        db_path=db_path or default_config.db_path,
        max_results=max_results or default_config.max_results,
    )
    web = Web(config)
    
    async def run():
        try:
            await run_workflow_run(web, workflow_id, vars)
        finally:
            await web.close()
    
    asyncio.run(run())


@workflow_app.command("list")
def workflow_list(
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Path to DuckDB database"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum search results"),
) -> None:
    """List all workflows."""
    default_config = WebConfig()
    config = WebConfig(
        db_path=db_path or default_config.db_path,
        max_results=max_results or default_config.max_results,
    )
    web = Web(config)
    
    async def run():
        try:
            await run_workflow_list(web)
        finally:
            await web.close()
    
    asyncio.run(run())


@workflow_app.command("show")
def workflow_show(
    workflow_id: str = typer.Argument(..., help="Workflow ID"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Path to DuckDB database"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum search results"),
) -> None:
    """Show workflow details."""
    default_config = WebConfig()
    config = WebConfig(
        db_path=db_path or default_config.db_path,
        max_results=max_results or default_config.max_results,
    )
    web = Web(config)
    
    async def run():
        try:
            await run_workflow_show(web, workflow_id)
        finally:
            await web.close()
    
    asyncio.run(run())


@workflow_app.command("autolearn")
def workflow_autolearn(
    base_url: str = typer.Argument(..., help="Base URL to learn"),
    workflow_id: Optional[str] = typer.Option(None, "--workflow-id", help="Workflow ID to improve (optional)"),
    iterations: int = typer.Option(5, "--iterations", help="Number of learning iterations"),
    db_path: Optional[str] = typer.Option(None, "--db-path", help="Path to DuckDB database"),
    max_results: Optional[int] = typer.Option(None, "--max-results", help="Maximum search results"),
) -> None:
    """Run auto-learning cycle."""
    default_config = WebConfig()
    config = WebConfig(
        db_path=db_path or default_config.db_path,
        max_results=max_results or default_config.max_results,
    )
    web = Web(config)
    
    async def run():
        try:
            await run_workflow_autolearn(web, base_url, workflow_id, iterations)
        finally:
            await web.close()
    
    asyncio.run(run())


if __name__ == "__main__":
    app()
