"""``semantica pipeline`` commands."""

from pathlib import Path
from typing import (List)
import yaml
import click
from ..utils.exceptions import SemanticaError
from . import _common


@_common.main.group(invoke_without_command=True)
@click.pass_context
def pipeline(ctx: click.Context) -> None:
    """Pipeline command group (foundation placeholder)."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@pipeline.command("init")
@click.option("--template", type=click.Choice(_common._PIPELINE_TEMPLATES),
              default="document_processing", show_default=True)
@click.option("--output", default="pipeline.yaml", show_default=True, type=click.Path())
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.pass_obj
def pipeline_init(cli_ctx: _common.CLIContext, template: str, output: str, local_dry: bool) -> None:
    """Scaffold a new pipeline config from a template.

    \b
    Example:
      semantica pipeline init --template document_processing --output pipeline.yaml
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "init pipeline", template=template, output=output)
            return
        try:
            from ..pipeline import PipelineSerializer, PipelineTemplateManager
        except ImportError as exc:
            raise click.ClickException(f"Pipeline module not available: {exc}") from exc
        builder = PipelineTemplateManager().create_pipeline_from_template(template)
        # serialize_pipeline(..., format="yaml") returns the dict form rather than
        # a YAML string, so the dump happens here.
        payload = PipelineSerializer().serialize_pipeline(
            builder.build(), format="yaml",
        )
        Path(output).write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8",
        )
        _common._ok(cli_ctx, f"Created {output} from template '{template}'")

    _common._run_with_error_handling(_action)


@pipeline.command("validate")
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def pipeline_validate(cli_ctx: _common.CLIContext, pipeline_file: str, local_json: bool) -> None:
    """Validate a pipeline config without running it."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        # --json promises its own {valid, errors, warnings, pipeline} object, so a
        # file the loader (or the builder inside it) rejects must still be reported
        # in that shape instead of falling through to the generic error envelope.
        json_mode = _common._is_json(cli_ctx, local_json)
        errors: List[str] = []
        warnings_found: List[str] = []
        pipeline = None
        try:
            pipeline = _common._load_pipeline_file(pipeline_file)
        except _common.PipelineFileError as exc:
            if not json_mode:
                raise
            errors = exc.errors
        except click.ClickException as exc:
            if not json_mode:
                raise
            errors = [str(exc.format_message())]
        except SemanticaError as exc:  # the builder rejected the payload
            if not json_mode:
                raise
            errors = _common._pipeline_builder_error_messages(exc)
        except Exception as exc:  # malformed in some way we did not anticipate
            if not json_mode:
                raise
            errors = [str(exc)]

        if pipeline is not None:
            try:
                from ..pipeline import PipelineValidator
            except ImportError as exc:
                raise click.ClickException(
                    f"Pipeline module not available: {exc}"
                ) from exc
            # The old wrapper passed the file path straight to validate(). A str has
            # no steps, so it type-checked nothing and reported valid for any input.
            result = PipelineValidator().validate(pipeline)
            errors = [str(e) for e in (getattr(result, "errors", None) or [])]
            warnings_found = [
                str(w) for w in (getattr(result, "warnings", None) or [])
            ]
            valid = bool(getattr(result, "valid", False))
        else:
            valid = False

        if json_mode:
            _common._jecho({
                "valid": valid,
                "errors": errors,
                "warnings": warnings_found,
                "pipeline": pipeline_file,
            })
            if not valid:
                # Text mode already exits non-zero on an invalid file; JSON mode
                # used to return 0, so callers had to parse the body to notice.
                raise SystemExit(1)
            return

        if not valid:
            for err in errors:
                _common._warn(cli_ctx, err)
            raise click.ClickException(
                f"Pipeline config invalid: {pipeline_file} "
                f"({len(errors)} error(s), {len(warnings_found)} warning(s))"
            )
        _common._ok(
            cli_ctx,
            f"Pipeline config valid: {pipeline_file} "
            f"({len(warnings_found)} warning(s))",
        )

    _common._run_with_error_handling(_action)


@pipeline.command("run")
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option("--parallel", default=4, type=int, show_default=True,
              help="Worker concurrency.")
@click.option("--incremental", is_flag=True, default=False,
              help="Delta mode — skip unchanged records.")
@click.option("--watch", is_flag=True, default=False,
              help="Re-run on source changes.")
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.pass_obj
def pipeline_run(cli_ctx: _common.CLIContext, pipeline_file: str, parallel: int,
                 incremental: bool, watch: bool, local_dry: bool) -> None:
    """Execute a pipeline.

    \b
    Example:
      semantica pipeline run pipeline.yaml --parallel 8
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "run pipeline", file=pipeline_file, parallel=parallel)
            return
        if watch:
            raise click.ClickException(
                "--watch has no implementation in the execution engine yet; "
                "re-run the pipeline after each change instead."
            )
        pipeline = _common._load_pipeline_file(pipeline_file)
        # _execute_step() returns its input untouched when step.handler is None, so
        # an all-unhandled pipeline "succeeds" having done nothing. Refuse instead
        # of reporting a green no-op.
        unhandled = [
            step.name for step in pipeline.steps
            if getattr(step, "handler", None) is None
        ]
        if unhandled:
            raise click.ClickException(
                f"Cannot execute {pipeline_file}: {len(unhandled)} step(s) have no "
                f"handler ({', '.join(unhandled)}). Step handlers are Python "
                "callables and are never restored from a config file, so the engine "
                "would pass data through every step and still report success. "
                "Register them with PipelineBuilder.register_step_handler() and run "
                "the pipeline in Python, or use `semantica pipeline validate` to "
                "check the config."
            )
        # --parallel and --incremental have no run-level switch in the engine:
        # parallelism is read from pipeline.config and delta mode is per step.
        pipeline.config["parallelism"] = parallel
        if incremental:
            for step in pipeline.steps:
                step.delta_mode = True
        try:
            from ..pipeline import ExecutionEngine
        except ImportError as exc:
            raise click.ClickException(f"Pipeline module not available: {exc}") from exc
        engine = ExecutionEngine(config=cli_ctx.config.to_dict())
        result = engine.execute_pipeline(pipeline)
        if not result.success:
            errors = [str(e) for e in (result.errors or [])]
            raise click.ClickException(
                f"Pipeline failed: {'; '.join(errors) if errors else 'no error detail'}"
            )
        elapsed = result.metadata.get("execution_time")
        pipeline_id = result.metadata.get("pipeline_id", pipeline.name)
        summary = f"Pipeline completed: {pipeline_id}"
        if isinstance(elapsed, (int, float)):
            summary += f" in {elapsed:.2f}s"
        _common._ok(cli_ctx, summary)

    _common._run_with_error_handling(_action)


@pipeline.command("status")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def pipeline_status(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Show status of a running pipeline."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        # ExecutionEngine.pipeline_status is an in-memory dict and
        # get_pipeline_status() only reads it, so a pipeline started by another
        # process is invisible here. There is no id to accept either: ids are
        # pipeline.name and are never persisted.
        raise click.ClickException(
            "`pipeline status` has no source of truth: the execution engine keeps "
            "pipeline status in memory, so a pipeline started by a separate "
            "`semantica pipeline run` process cannot be observed from here. Track it "
            "in the process that called ExecutionEngine.execute_pipeline() instead."
        )

    _common._run_with_error_handling(_action)


@pipeline.command("stop")
@click.pass_obj
def pipeline_stop(cli_ctx: _common.CLIContext) -> None:
    """Stop a running pipeline."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        # stop_pipeline() does set a flag the run loop honours (execution_engine.py
        # :273 and :320), but it takes a pipeline id and that flag lives on the
        # engine instance doing the stopping, so it cannot reach a pipeline owned by
        # another process.
        raise click.ClickException(
            "`pipeline stop` has nothing to stop: the execution engine tracks running "
            "pipelines in memory only, so a pipeline started by a separate "
            "`semantica pipeline run` process is not reachable from here. Call "
            "ExecutionEngine.stop_pipeline(pipeline_id) in the process running the "
            "pipeline instead."
        )

    _common._run_with_error_handling(_action)
