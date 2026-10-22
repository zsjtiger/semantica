import { useState } from "react";
import { GitMerge, ArrowRight, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import i18next from "i18next";
import { logEvent } from "../../store/registryStore";

interface FieldRow { labelKey: string; primary: string; duplicate: string; differs: boolean }

const MOCK_FIELDS: FieldRow[] = [
  { labelKey: "dataTools.diffMerge.fields.name",    primary: "Sample Company Inc.", duplicate: "Sample Company",    differs: true  },
  { labelKey: "dataTools.diffMerge.fields.founded", primary: "2004-05-12",          duplicate: "2004-05-12",        differs: false },
  { labelKey: "dataTools.diffMerge.fields.type",    primary: "Organization",        duplicate: "Organisation",      differs: true  },
  { labelKey: "dataTools.diffMerge.fields.country", primary: "US",                  duplicate: "US",                differs: false },
];

export function DiffMergeWorkspace() {
  const { t } = useTranslation();
  const [primaryId, setPrimaryId]   = useState("n-primary-1");
  const [duplicateId, setDuplicateId] = useState("n-dup-2");
  const [status, setStatus]   = useState<"idle" | "loading" | "success" | "error">("idle");
  const [msg, setMsg]         = useState("");

  async function handleMerge() {
    setStatus("loading");
    setMsg("");
    try {
      const res = await fetch("/api/enrich/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ primary_id: primaryId, duplicate_ids: [duplicateId] }),
      });
      const data = await res.json();
      if (data.merged_into) {
        setStatus("success");
        setMsg(t('dataTools.diffMerge.merged', { target: data.merged_into, count: data.edges_updated ?? 0 }));
        logEvent("merge", i18next.t('dataTools.diffMerge.mergedLog', { duplicate: duplicateId, target: data.merged_into, count: data.edges_updated ?? 0 }), {
          primary: data.merged_into, duplicate: duplicateId, edgesUpdated: data.edges_updated,
        });
      } else {
        throw new Error(data.detail || t('dataTools.diffMerge.unexpectedResponse'));
      }
    } catch (e: unknown) {
      setStatus("error");
      setMsg(e instanceof Error ? e.message : t('dataTools.diffMerge.mergeFailed'));
    }
  }

  return (
    <div className="ws-page ws-scroll">
      <div className="ws-padded" style={{ display: "flex", flexDirection: "column", gap: 22, maxWidth: 1000, margin: "0 auto", width: "100%" }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ width: 42, height: 42, borderRadius: 13, background: "var(--ws-purple-soft)", border: "1px solid rgba(192,132,252,0.3)", display: "grid", placeItems: "center", color: "var(--ws-purple)", flexShrink: 0 }}>
            <GitMerge size={20} />
          </div>
          <div>
            <h2 className="ws-title" style={{ fontSize: 18 }}>{t('dataTools.diffMerge.title')}</h2>
            <div className="ws-body" style={{ marginTop: 2 }}>{t('dataTools.diffMerge.subtitle')}</div>
          </div>
        </div>

        {/* ID inputs */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 12, alignItems: "end" }}>
          <div>
            <label className="ws-label">{t('dataTools.diffMerge.primaryLabel')}</label>
            <input className="ws-input" value={primaryId} onChange={(e) => { setPrimaryId(e.target.value); setStatus("idle"); }} placeholder="e.g. n-primary-1" />
          </div>
          <div style={{ paddingBottom: 2, color: "var(--ws-text-dim)" }}>
            <ArrowRight size={18} />
          </div>
          <div>
            <label className="ws-label">{t('dataTools.diffMerge.duplicateLabel')}</label>
            <input className="ws-input" value={duplicateId} onChange={(e) => { setDuplicateId(e.target.value); setStatus("idle"); }} placeholder="e.g. n-dup-2" />
          </div>
        </div>

        {/* Diff table */}
        <div className="ws-card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ padding: "6px 16px", background: "rgba(242,182,109,0.06)", borderBottom: "1px solid rgba(242,182,109,0.15)", display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 10, fontWeight: 700, color: "var(--ws-amber)", letterSpacing: "0.08em", textTransform: "uppercase" }}>{t('dataTools.diffMerge.sampleBadge')}</span>
            <span style={{ fontSize: 11, color: "var(--ws-text-dim)" }}>{t('dataTools.diffMerge.sampleNote')}</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "140px 1fr 1fr", background: "rgba(0,0,0,0.28)", borderBottom: "1px solid var(--ws-border)" }}>
            <div style={{ padding: "10px 16px", fontSize: 11, fontWeight: 700, color: "var(--ws-text-dim)", letterSpacing: "0.08em", textTransform: "uppercase" }}>{t('dataTools.diffMerge.colField')}</div>
            <div style={{ padding: "10px 16px", fontSize: 11, fontWeight: 700, color: "var(--ws-accent)", letterSpacing: "0.08em", textTransform: "uppercase", borderLeft: "1px solid var(--ws-border)" }}>{t('dataTools.diffMerge.colPrimary')}</div>
            <div style={{ padding: "10px 16px", fontSize: 11, fontWeight: 700, color: "#fca5a5", letterSpacing: "0.08em", textTransform: "uppercase", borderLeft: "1px solid var(--ws-border)" }}>{t('dataTools.diffMerge.colDuplicate')}</div>
          </div>
          {MOCK_FIELDS.map((row) => (
            <div key={row.labelKey} style={{ display: "grid", gridTemplateColumns: "140px 1fr 1fr", borderBottom: "1px solid rgba(74,163,255,0.06)", background: row.differs ? "rgba(242,182,109,0.03)" : "transparent" }}>
              <div style={{ padding: "12px 16px", fontSize: 12, fontWeight: 600, color: "var(--ws-text-dim)" }}>{t(row.labelKey)}</div>
              <div style={{ padding: "12px 16px", fontSize: 13, color: "var(--ws-text)", borderLeft: "1px solid var(--ws-border)" }}>{row.primary}</div>
              <div style={{ padding: "12px 16px", fontSize: 13, color: row.differs ? "#fbbf24" : "var(--ws-text)", fontWeight: row.differs ? 700 : 400, borderLeft: "1px solid var(--ws-border)" }}>
                {row.duplicate}
                {row.differs && <span className="ws-pill ws-pill--amber" style={{ marginLeft: 8, fontSize: 9 }}>{t('dataTools.diffMerge.diffBadge')}</span>}
              </div>
            </div>
          ))}
        </div>

        {/* Status & action */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {status === "success" && (
            <div style={{ display: "flex", alignItems: "center", gap: 7, color: "#6ee7b7", fontSize: 13 }}>
              <CheckCircle2 size={15} />{msg}
            </div>
          )}
          {status === "error" && (
            <div style={{ display: "flex", alignItems: "center", gap: 7, color: "#fca5a5", fontSize: 13 }}>
              <AlertCircle size={15} />{msg}
            </div>
          )}
          <button
            className="ws-btn ws-btn--primary"
            onClick={handleMerge}
            disabled={status === "loading" || !primaryId || !duplicateId}
            style={{ marginLeft: "auto" }}
          >
            {status === "loading" ? <><Loader2 size={14} className="ws-spin" />{t('dataTools.diffMerge.merging')}</> : <><GitMerge size={14} />{t('dataTools.diffMerge.confirmMerge')}</>}
          </button>
        </div>
      </div>
    </div>
  );
}
