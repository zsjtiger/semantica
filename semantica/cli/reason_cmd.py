"""``semantica reason`` commands."""

from typing import (Any, Dict, List,
                    Optional)
import click
from rich.table import Table
from . import _common


# ─── Intelligence ─────────────────────────────────────────────────────────────


@_common.main.group(invoke_without_command=True)
@click.pass_context
def reason(ctx: click.Context) -> None:
    """Run reasoning engines and explain conclusions."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@reason.command("run")
@click.option("--engine",
              type=click.Choice(["deductive", "abductive", "rete", "forward-chain",
                                  "datalog", "sparql", "graph"]),
              default="rete", show_default=True)
@click.option("--rules", default=None, type=click.Path(exists=True),
              help="Custom rules file (YAML for rete, Datalog Horn clauses for datalog).")
@click.option("--query", "query_text", default=None,
              help="Natural-language question, required for --engine graph.")
@click.option("--premises", "premises_file", default=None, type=click.Path(exists=True),
              help="Premises file for --engine deductive (YAML list of statements; "
                   "falls back to graph-store facts if omitted).")
@click.option("--observations", "observations_file", default=None, type=click.Path(exists=True),
              help="Observations file for --engine abductive (required).")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def reason_run(cli_ctx: _common.CLIContext, engine: str, rules: Optional[str],
               query_text: Optional[str], premises_file: Optional[str],
               observations_file: Optional[str], local_json: bool) -> None:
    """Execute a reasoning engine against the knowledge graph.

    \b
    Example:
      semantica reason run --engine rete --rules business-rules.yaml
      semantica reason run --engine datalog --rules facts.dl
      semantica reason run --engine graph --query "Who manages Bob?"
      semantica reason run --engine deductive --premises premises.yaml --rules rules.yaml
      semantica reason run --engine abductive --observations observations.yaml --rules rules.yaml
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        if engine in ("rete", "forward-chain"):
            try:
                from ..reasoning import Reasoner
                # Reasoner has no run() method (#1354); dispatch to its real
                # API: facts from the configured graph store + rules from the
                # optional --rules file into infer_facts().
                r = Reasoner(engine=engine, config=cli_ctx.config.to_dict())
                rule_defs = _common._load_rule_definitions(rules) if rules else None
                facts = _common._graph_store_facts(cli_ctx)

                def _infer() -> Dict[str, Any]:
                    inferred = r.infer_facts(facts, rule_defs)
                    return {
                        "engine": engine,
                        "facts": len(facts),
                        "inferred_count": len(inferred),
                        "inferred_facts": inferred,
                    }

                result = _common._run_reasoning_with_status(cli_ctx, engine, local_json, _infer)
            except ImportError as exc:
                raise click.ClickException(f"Reasoning module not available: {exc}") from exc

        elif engine == "datalog":
            try:
                from ..reasoning import DatalogReasoner
                dr = DatalogReasoner(config=cli_ctx.config.to_dict())
                for rule_def in (_common._load_rule_definitions(rules) if rules else []):
                    dr.add_rule(rule_def)
                # Datalog reads a leading-uppercase *argument* as a variable
                # (see datalog_reasoner.py _is_variable(); predicate case is
                # unconstrained), so graph-store facts like "Person(Alice)"
                # need their arguments -- not the predicate -- lowercased to
                # read as ground constants. Lowercasing the whole string
                # would also break matching against rules written against
                # the graph's own (typically title-case) label spelling,
                # e.g. "Human(X) :- Person(X)."
                fact_strings = [_common._lowercase_datalog_args(f) for f in _common._graph_store_facts(cli_ctx)]
                for fact_str in fact_strings:
                    dr.add_fact(fact_str)

                def _infer() -> Dict[str, Any]:
                    before = set(fact_strings)
                    derived = dr.derive_all()
                    inferred = [f for f in derived if f not in before]
                    return {
                        "engine": engine,
                        "facts": len(fact_strings),
                        "inferred_count": len(inferred),
                        "inferred_facts": inferred,
                    }

                result = _common._run_reasoning_with_status(cli_ctx, engine, local_json, _infer)
            except ImportError as exc:
                raise click.ClickException(f"Reasoning module not available: {exc}") from exc

        elif engine == "graph":
            if not query_text:
                raise click.ClickException(
                    "Engine 'graph' requires --query \"<question>\".")
            try:
                from ..reasoning import GraphReasoner
                gr = GraphReasoner(config=cli_ctx.config.to_dict())
                graph = _common._graph_store_as_context(cli_ctx)

                def _infer() -> Dict[str, Any]:
                    answer = gr.reason(graph, query_text)
                    # GraphReasoner.reason() never raises on an LLM-side
                    # failure (no provider configured, generation error) --
                    # it returns a string starting with "Error" instead.
                    # Surface that as a real command failure (non-zero exit)
                    # rather than a successful result an automated caller
                    # would read as a real answer.
                    if answer.startswith("Error: LLM provider not initialized") or \
                            answer.startswith("Error during reasoning:"):
                        raise click.ClickException(answer)
                    return {
                        "engine": engine,
                        "query": query_text,
                        "facts": len(graph["entities"]) + len(graph["relationships"]),
                        "answer": answer,
                    }

                result = _common._run_reasoning_with_status(cli_ctx, engine, local_json, _infer)
            except ImportError as exc:
                raise click.ClickException(f"Reasoning module not available: {exc}") from exc

        elif engine == "deductive":
            try:
                # DeductiveReasoner/Premise aren't re-exported from
                # semantica.reasoning's __init__ (unlike DatalogReasoner/
                # GraphReasoner) -- import from the submodule directly.
                from ..reasoning.deductive_reasoner import DeductiveReasoner, Premise
                dr = DeductiveReasoner(config=cli_ctx.config.to_dict())
                for rule_def in (_common._load_rule_definitions(rules) if rules else []):
                    dr.reasoner.add_rule(rule_def)
                # DeductiveReasoner uses the same "IF X THEN Y" / "?x"
                # syntax as rete (both go through Reasoner.add_rule()), so
                # --rules files are shared across the two engines unchanged.
                if premises_file:
                    premises = _common._load_premises(premises_file)
                else:
                    premises = [Premise(premise_id=f"fact_{i}", statement=fact)
                                for i, fact in enumerate(_common._graph_store_facts(cli_ctx))]

                def _infer() -> Dict[str, Any]:
                    # apply_logic() scans self.reasoner.rules once, not to a
                    # fixpoint -- a conclusion whose prerequisite premise is
                    # itself a *later* rule's conclusion would be silently
                    # dropped from a single call. known_facts persists on
                    # the DeductiveReasoner instance across calls (premises
                    # are just re-added, which is a no-op on a set), so
                    # looping until a call yields nothing new reaches the
                    # same fixpoint infer_facts()/derive_all() reach for
                    # the other engines.
                    conclusions: List[Any] = []
                    while True:
                        new_conclusions = dr.apply_logic(premises)
                        if not new_conclusions:
                            break
                        conclusions.extend(new_conclusions)
                    return {
                        "engine": engine,
                        "facts": len(premises),
                        "inferred_count": len(conclusions),
                        "inferred_facts": [c.statement for c in conclusions],
                    }

                result = _common._run_reasoning_with_status(cli_ctx, engine, local_json, _infer)
            except ImportError as exc:
                raise click.ClickException(f"Reasoning module not available: {exc}") from exc

        elif engine == "abductive":
            if not observations_file:
                raise click.ClickException(
                    "Engine 'abductive' requires --observations <file>.")
            try:
                from ..reasoning.abductive_reasoner import AbductiveReasoner
                ab = AbductiveReasoner(config=cli_ctx.config.to_dict())
                for rule_def in (_common._load_rule_definitions(rules) if rules else []):
                    ab.reasoner.add_rule(rule_def)
                # Deliberately not calling _graph_store_facts()/
                # add_knowledge() here: AbductiveReasoner's current
                # generate_hypotheses()/find_explanations() never reads
                # knowledge_base, so it would have no effect on the result
                # while forcing every abductive run to depend on the
                # configured graph store being reachable -- even a fully
                # self-contained --rules + --observations run would fail if
                # the graph backend is down. Revisit once knowledge_base is
                # actually consumed by the ranking/generation algorithm.
                observations = _common._load_observations(observations_file)

                def _infer() -> Dict[str, Any]:
                    explanations = ab.find_explanations(observations)
                    return {
                        "engine": engine,
                        "observations": len(observations),
                        "explanations": [
                            {
                                "observation": exp.observation.description,
                                "best_hypothesis": (
                                    exp.best_hypothesis.explanation
                                    if exp.best_hypothesis else None
                                ),
                                "confidence": exp.confidence,
                                "hypotheses_considered": len(exp.hypotheses),
                            }
                            for exp in explanations
                        ],
                    }

                result = _common._run_reasoning_with_status(cli_ctx, engine, local_json, _infer)
            except ImportError as exc:
                raise click.ClickException(f"Reasoning module not available: {exc}") from exc

        else:
            # sparql is the only remaining engine choice, and it has no
            # execution path to dispatch to at all: SPARQLReasoner.
            # execute_query() raises NotImplementedError unconditionally
            # and the class has no .query() method either. Fail honestly
            # instead of silently forward-chaining under another engine's
            # name.
            raise click.ClickException(
                "Engine 'sparql' is not wired to 'reason run' yet; "
                "supported engines: rete, forward-chain, datalog, graph, "
                "deductive, abductive. Use 'semantica reason query' for "
                "SPARQL queries.")

        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, dict) else {"result": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@reason.command("explain")
@click.argument("conclusion")
@click.option("--depth", default=3, type=int, show_default=True)
@click.option("--format", "fmt", type=click.Choice(["text", "markdown", "json"]),
              default="text", show_default=True)
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def reason_explain(cli_ctx: _common.CLIContext, conclusion: str, depth: int,
                   fmt: str, local_json: bool) -> None:
    """Explain how a conclusion was reached.

    \b
    Example:
      semantica reason explain "Alice is-manager-of Engineering" --format markdown
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..reasoning import ExplanationGenerator
            gen = ExplanationGenerator(config=cli_ctx.config.to_dict())
            if cli_ctx.quiet or cli_ctx.json_output:
                expl = gen.explain(conclusion, depth=depth)
            else:
                with _common.console.status(
                    f"[{_common._DIM}]Generating explanation (depth={depth})…[/{_common._DIM}]",
                    spinner="dots",
                ):
                    expl = gen.explain(conclusion, depth=depth)
        except ImportError as exc:
            raise click.ClickException(f"Reasoning module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json) or fmt == "json":
            _common._jecho(expl if isinstance(expl, dict) else {"explanation": str(expl)})
        else:
            _common._pprint(cli_ctx, expl)

    _common._run_with_error_handling(_action)


@reason.command("query")
@click.argument("query_str")
@click.option("--with-inference", is_flag=True, default=False,
              help="Include inferred facts in results.")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def reason_query(cli_ctx: _common.CLIContext, query_str: str, with_inference: bool,
                 local_json: bool) -> None:
    """SPARQL or Datalog query with inference.

    \b
    Example:
      semantica reason query "SELECT ?x WHERE { ?x :hasRole :Manager }" --with-inference
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        try:
            from ..reasoning import SPARQLReasoner
            r = SPARQLReasoner(config=cli_ctx.config.to_dict())
            result = r.query(query_str, with_inference=with_inference)
        except ImportError as exc:
            raise click.ClickException(f"Reasoning module not available: {exc}") from exc
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(result if isinstance(result, (dict, list)) else {"result": str(result)})
        else:
            _common._pprint(cli_ctx, result)

    _common._run_with_error_handling(_action)


@reason.command("list")
@click.pass_obj
def reason_list(cli_ctx: _common.CLIContext) -> None:
    """List available reasoning engines."""
    cli_ctx = _common._require_ctx(cli_ctx)
    _builtin: List[str] = ["deductive", "abductive", "rete", "forward-chain", "datalog", "sparql", "graph"]
    try:
        import semantica.reasoning as _reasoning_mod
        _fn = getattr(_reasoning_mod, "get_available_engines", None)
        engines: List[str] = list(_fn()) if callable(_fn) else _builtin
    except ImportError:
        engines = _builtin
    if cli_ctx.json_output:
        _common._jecho({"engines": engines})
    else:
        table = Table(title="[bold]Available Reasoning Engines[/bold]",
                      box=_common._TABLE_BOX, show_edge=False, padding=(0, 1))
        table.add_column("Engine", style=_common._KEY)
        for e in engines:
            table.add_row(e)
        _common.console.print(table)
