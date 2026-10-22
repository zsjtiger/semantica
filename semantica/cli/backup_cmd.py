"""``semantica backup`` commands."""

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import (Any, Dict, Optional)
import click
from rich.table import Table
from .. import __version__
from . import _common


# ─── Backup ───────────────────────────────────────────────────────────────────


@_common.main.group(invoke_without_command=True)
@click.pass_context
def backup(ctx: click.Context) -> None:
    """Back up and restore all Semantica data."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@backup.command("info")
@click.option("--json", "local_json", is_flag=True, default=False)
@click.pass_obj
def backup_info(cli_ctx: _common.CLIContext, local_json: bool) -> None:
    """Show backend type, data location, and backup method per store."""
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        cfg = cli_ctx.config.to_dict()
        # Config normalizes store sections into graph_db / vector_store / triplet_store
        section_keys = {
            "graph": "graph_db",
            "vector": "vector_store",
            "triplet": "triplet_store",
        }
        rows = []
        for store_type, cfg_key in section_keys.items():
            info = cfg.get(cfg_key)
            if not isinstance(info, dict) or not info:
                continue
            backend = info.get("backend", "unknown")
            uri_raw = info.get("uri", info.get("host", ""))
            uri = _common._redact(str(uri_raw)) if uri_raw else "(none)"
            cloud_backends = {"pinecone", "neptune", "snowflake"}
            if backend in cloud_backends:
                method = "use `semantica export` (cloud-managed)"
            elif backend in {"neo4j", "qdrant", "blazegraph", "falkordb"}:
                method = "native dump API"
            else:
                method = "file copy"
            rows.append({"store": store_type, "backend": backend, "uri": uri, "method": method})
        if _common._is_json(cli_ctx, local_json):
            _common._jecho(rows)
        else:
            table = Table(title="[bold]Backup Info[/bold] [dim](credentials redacted)[/dim]",
                          box=_common._TABLE_BOX, show_edge=False, padding=(0, 1))
            table.add_column("Store", style=_common._KEY, no_wrap=True)
            table.add_column("Backend")
            table.add_column("URI/Location")
            table.add_column("Method", style=_common._VAL)
            for r in rows:
                table.add_row(r["store"], r["backend"], r["uri"], r["method"])
            _common.console.print(table)

    _common._run_with_error_handling(_action)


@backup.command("create")
@click.argument("destination", type=click.Path())
@click.option("--include",
              type=click.Choice(["graph", "vector", "triplet", "config", "ontology", "all"]),
              default="all", show_default=True)
@click.option("--compress", is_flag=True, default=False)
@click.option("--encrypt", is_flag=True, default=False,
              help="AES-256 encrypt (passphrase prompted interactively).")
@click.option("--keyfile", default=None, type=click.Path(),
              help="Read passphrase from file (must not be world-readable).")
@click.option("--strip-config", is_flag=True, default=False,
              help="Exclude semantica.yaml to avoid storing credentials at rest.")
@click.option("--fail-on-skip", is_flag=True, default=False,
              help="Exit non-zero when a configured store path does not exist.")
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.option("--quiet", "local_quiet", is_flag=True, default=False)
@click.pass_obj
def backup_create(
    cli_ctx: _common.CLIContext, destination: str, include: str, compress: bool,
    encrypt: bool, keyfile: Optional[str], strip_config: bool,
    fail_on_skip: bool, local_dry: bool, local_quiet: bool,
) -> None:
    """Create a full backup archive.

    \b
    Example:
      semantica backup create ./backups/semantica-2026-05-25.tar.gz --compress --encrypt
    """
    cli_ctx = _common._require_ctx(cli_ctx)
    quiet = cli_ctx.quiet or local_quiet

    def _action() -> None:
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "backup create", destination=destination, include=include)
            return

        if keyfile:
            kp = Path(keyfile)
            if not kp.exists():
                raise click.ClickException(f"Keyfile not found: {keyfile}")
            stat_result = kp.stat()
            # Reject group-readable (0o040) or world-readable (0o004) keyfiles.
            if stat_result.st_mode & 0o044:
                raise click.ClickException(
                    f"Keyfile {keyfile} is readable by group or others. "
                    f"Run: chmod 600 {keyfile}"
                )
            # On POSIX, also verify ownership to catch sudo/setuid misuse.
            if hasattr(os, "getuid") and stat_result.st_uid != os.getuid():  # type: ignore[attr-defined]
                raise click.ClickException(
                    f"Keyfile {keyfile} is not owned by the current user."
                )
            passphrase = kp.read_text(encoding="utf-8").strip()
            if not passphrase:
                raise click.ClickException(f"Keyfile {keyfile} is empty; cannot derive encryption key.")
        else:
            passphrase = None

        cfg = cli_ctx.config.to_dict()
        sources, skipped = _common._collect_backup_sources(
            cfg, cli_ctx.config_path, include, strip_config,
        )
        # A configured store whose path is missing was dropped without a word,
        # so a partial archive was reported as a success (#2009). Report every
        # one before the prompts, and before the empty-archive check below so
        # the missing paths are named even when no store is left to back up.
        for entry in skipped:
            _common._warn(
                cli_ctx,
                f"Skipping {entry['store']} store ({entry['backend']}): "
                f"path does not exist: {entry['path']}",
            )

        # Fail before any prompt instead of writing an archive with no
        # store data in it (#1820).
        if include in ("graph", "vector", "triplet", "all") and not any(
            arcname.startswith(("graph/", "vector/", "triplet/"))
            for arcname, _ in sources
        ):
            stores = "graph, vector and triplet stores" if include == "all" else f"{include} store"
            raise click.ClickException(
                f"No store data to back up: found no local files for the {stores}. "
                "backup create only copies file-based stores (a graph_db / vector_store / "
                "triplet_store path that exists on disk); run `semantica backup info` "
                "to see how to back up other backends."
            )

        if skipped and fail_on_skip:
            missing = ", ".join(f"{e['store']} ({e['path']})" for e in skipped)
            raise click.ClickException(
                f"Configured store path does not exist: {missing}. Nothing from "
                "those stores would be in the archive; create the path, drop it "
                "from the config, or omit --fail-on-skip to back up the rest."
            )

        if encrypt and passphrase is None:
            passphrase = click.prompt("Backup passphrase", hide_input=True,
                                      confirmation_prompt=True)

        if not strip_config and not encrypt and passphrase is None:
            if not quiet:
                _common.console.print(
                    f"[{_common._WARN_STY}] ⚠[/{_common._WARN_STY}] backup includes semantica.yaml which may contain "
                    "credentials. Use --encrypt or --strip-config to suppress this warning."
                )
            click.confirm("Continue without encryption?", abort=True)

        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)

        import tarfile, tempfile, shutil
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest: Dict[str, Any] = {
                "include": include,
                "version": str(__version__),
                "files": [arcname for arcname, _ in sources],
                "skipped": skipped,
            }
            (tmp_path / "backup_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8",
            )
            # Build the tar archive in a temp file; we'll compress/encrypt on top.
            raw_tar = Path(tmp) / "semantica-backup.tar"
            with tarfile.open(str(raw_tar), "w") as tar:
                tar.add(str(tmp_path / "backup_manifest.json"),
                        arcname="semantica-backup/backup_manifest.json")
                for arcname, local_path in sources:
                    tar.add(str(local_path), arcname=f"semantica-backup/{arcname}")

            # Compress (gzip) if requested.
            if compress:
                import gzip
                compressed = Path(tmp) / "semantica-backup.tar.gz"
                with open(str(raw_tar), "rb") as f_in, gzip.open(str(compressed), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                raw_tar.unlink()
                payload_path = compressed
            else:
                payload_path = raw_tar

            # Encrypt with AES-256-GCM if --encrypt was requested.
            if encrypt and passphrase:
                try:
                    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                    from cryptography.hazmat.primitives import hashes
                    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                except ImportError as exc:
                    raise click.ClickException(
                        "Encryption requires the 'cryptography' package: "
                        "pip install cryptography"
                    ) from exc
                salt = os.urandom(32)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(), length=32,
                    salt=salt, iterations=480_000,
                )
                key = kdf.derive(passphrase.encode())
                nonce = os.urandom(12)
                plaintext = payload_path.read_bytes()
                ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
                # Wire format: magic(4) + salt(32) + nonce(12) + ciphertext
                enc_path = Path(tmp) / "semantica-backup.enc"
                enc_path.write_bytes(b"SEM1" + salt + nonce + ciphertext)
                payload_path = enc_path

            shutil.copy2(str(payload_path), str(dest))

        try:
            os.chmod(str(dest), 0o600)
        except OSError:
            pass

        if not quiet:
            enc_note = " (AES-256-GCM encrypted)" if encrypt and passphrase else ""
            _common._ok(cli_ctx, f"Backup created: {destination}{enc_note}")

    _common._run_with_error_handling(_action)


@backup.command("sync")
@click.argument("destination", type=click.Path())
@click.option("--include",
              type=click.Choice(["graph", "vector", "triplet", "config", "ontology", "all"]),
              default="all", show_default=True)
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.option("--quiet", "local_quiet", is_flag=True, default=False)
@click.pass_obj
def backup_sync(cli_ctx: _common.CLIContext, destination: str, include: str,
                local_dry: bool, local_quiet: bool) -> None:
    """Incremental folder sync (idempotent, copies only changed files).

    \b
    Example:
      semantica backup sync /mnt/external/semantica-backup
    """
    cli_ctx = _common._require_ctx(cli_ctx)
    quiet = cli_ctx.quiet or local_quiet

    def _action() -> None:
        import shutil
        dest = Path(destination)
        cfg = cli_ctx.config.to_dict()
        sources, _skipped = _common._collect_backup_sources(
            cfg, cli_ctx.config_path, include, strip_config=False,
        )
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "sync backup", destination=str(dest), include=include,
                 files=[a for a, _ in sources])
            return
        dest.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(dest), 0o700)
        except OSError:
            pass
        copied = 0
        for arcname, src_path in sources:
            dst_file = dest / arcname
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            # Incremental: skip if destination is up-to-date
            if dst_file.exists() and dst_file.stat().st_mtime >= src_path.stat().st_mtime:
                continue
            shutil.copy2(str(src_path), str(dst_file))
            copied += 1
        if not quiet:
            _common._ok(cli_ctx, f"Synced {copied} file(s) to {destination}")

    _common._run_with_error_handling(_action)


@backup.command("restore")
@click.argument("source", type=click.Path(exists=True))
@click.option("--dry-run", "local_dry", is_flag=True, default=False)
@click.pass_obj
def backup_restore(cli_ctx: _common.CLIContext, source: str, local_dry: bool) -> None:
    """Restore from a backup archive or folder.

    \b
    Example:
      semantica backup restore ./backups/semantica-2026-05-25.tar.gz --dry-run
    """
    cli_ctx = _common._require_ctx(cli_ctx)

    def _action() -> None:
        import shutil, tarfile as _tf
        src = Path(source)
        if _common._is_dry(cli_ctx, local_dry):
            _common._dry(cli_ctx, "restore", source=str(src))
            return

        # Determine payload: decrypt .enc first, then extract archive or copy dir.
        work_path = src
        if src.suffix == ".enc":
            passphrase = click.prompt("Backup passphrase", hide_input=True)
            try:
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            except ImportError as exc:
                raise click.ClickException(
                    "Decryption requires the 'cryptography' package: pip install cryptography"
                ) from exc
            raw = src.read_bytes()
            if not raw.startswith(b"SEM1") or len(raw) < 4 + 32 + 12:
                raise click.ClickException("Unrecognised or corrupted backup file.")
            salt, nonce, ciphertext = raw[4:36], raw[36:48], raw[48:]
            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480_000)
            key = kdf.derive(passphrase.encode())
            try:
                plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
            except Exception as exc:
                raise click.ClickException("Decryption failed — wrong passphrase?") from exc
            import tempfile
            tmp_enc = tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz")
            tmp_enc.write(plaintext)
            tmp_enc.close()
            work_path = Path(tmp_enc.name)

        try:
            if _tf.is_tarfile(str(work_path)):
                restore_root = Path.cwd().resolve()
                with _tf.open(str(work_path), "r:*") as tar:
                    # Dry-run listing was already handled above; extract now
                    for member in tar.getmembers():
                        # Strip the leading "semantica-backup/" prefix
                        member.name = member.name.replace("semantica-backup/", "", 1)
                        if not member.name:
                            continue

                        # Reject members whose resolved path escapes the
                        # restore root (path traversal / absolute paths),
                        # regardless of the "semantica-backup/" prefix.
                        member_path = (restore_root / member.name).resolve()
                        try:
                            member_path.relative_to(restore_root)
                        except ValueError:
                            raise click.ClickException(
                                f"Refusing to restore '{member.name}': "
                                "path escapes the restore directory."
                            )

                        # Reject symlink/hardlink members whose target
                        # escapes the restore root. Checked two ways:
                        # lexically (linkname itself, so an absolute path or
                        # a literal ".." segment is rejected outright, with
                        # no dependence on what else does or doesn't already
                        # exist on disk) and by resolution (catches any
                        # remaining traversal the lexical check misses).
                        if member.issym() or member.islnk():
                            linkname = member.linkname or ""
                            linkname_parts = PurePosixPath(
                                linkname.replace("\\", "/")
                            ).parts
                            if (
                                not linkname
                                or os.path.isabs(linkname)
                                or PureWindowsPath(linkname).is_absolute()
                                or ".." in linkname_parts
                            ):
                                raise click.ClickException(
                                    f"Refusing to restore '{member.name}': "
                                    "link target is absolute or traverses "
                                    "out of the archive."
                                )
                            link_target = (
                                member_path.parent / linkname
                            ).resolve()
                            try:
                                link_target.relative_to(restore_root)
                            except ValueError:
                                raise click.ClickException(
                                    f"Refusing to restore '{member.name}': "
                                    "link target escapes the restore directory."
                                )

                        extract_kwargs: Dict[str, Any] = {"path": str(restore_root)}
                        if hasattr(_tf, "data_filter"):
                            # Python >=3.12: also reject device files, and
                            # further harden the traversal/ownership checks.
                            extract_kwargs["filter"] = "data"
                        tar.extract(member, **extract_kwargs)
                        _common.console.print(f"  restored: {member.name}")
            elif src.is_dir():
                restore_root = Path.cwd()
                for f in src.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(src)
                        dst = restore_root / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(f), str(dst))
                        _common.console.print(f"  restored: {rel}")
            else:
                raise click.ClickException(f"Cannot restore from {source}: not a tar archive or directory.")
        finally:
            if src.suffix == ".enc":
                work_path.unlink(missing_ok=True)

        _common._ok(cli_ctx, f"Restore complete from {source}")

    _common._run_with_error_handling(_action)


@backup.command("schedule")
@click.option("--dest", required=True, type=click.Path(), help="Backup destination path.")
@click.option("--freq", type=click.Choice(["daily", "hourly", "weekly"]),
              default="daily", show_default=True)
@click.option("--encrypt", is_flag=True, default=False)
@click.pass_obj
def backup_schedule(cli_ctx: _common.CLIContext, dest: str, freq: str, encrypt: bool) -> None:
    """Print a ready-to-paste cron snippet.

    \b
    Example:
      semantica backup schedule --dest /mnt/external/semantica --freq daily --encrypt
    """
    cli_ctx = _common._require_ctx(cli_ctx)
    schedules = {"daily": "0 2 * * *", "hourly": "0 * * * *", "weekly": "0 2 * * 0"}
    cron_time = schedules[freq]
    enc_flag = " --encrypt" if encrypt else ""
    snippet = (
        f"{cron_time} semantica backup sync {dest}{enc_flag} --quiet"
    )
    if cli_ctx.json_output:
        _common._jecho({"cron": snippet})
    else:
        _common.console.print(f"[bold]Suggested cron entry:[/bold]\n{snippet}")
