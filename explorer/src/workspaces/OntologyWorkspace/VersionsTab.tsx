import { useTranslation } from "react-i18next";
import { useCallback, useEffect, useState } from "react";
import i18next from "i18next";
import {
  Layers,
  GitMerge,
  Clock,
  FileText,
  CheckCircle,
  XCircle,
  AlertCircle,
  Send,
  X,
  ArrowRight,
  Scale,
} from "lucide-react";

interface VersionEntry {
  version_id: string;
  ontology_uri: string;
  state: "draft" | "published";
  author: string;
  date: string;
  diff_summary: Record<string, any>;
}

interface Proposal {
  proposal_id: string;
  draft_id: string;
  ontology_uri: string;
  summary: string;
  author: string;
  reviewer: string | null;
  state: "draft" | "proposed" | "approved" | "published" | "rejected";
  impact_analysis: Record<string, any>;
  shacl_validation: Record<string, any>;
  created_at: string;
  updated_at: string;
  comments: Record<string, any>[];
}

export function VersionsTab() {
  const { t } = useTranslation();
  const [ontologyUri, setOntologyUri] = useState<string>("");
  const [versions, setVersions] = useState<VersionEntry[]>([]);
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [selectedProposal, setSelectedProposal] = useState<Proposal | null>(null);
  const [showProposalModal, setShowProposalModal] = useState(false);
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [comparePair, setComparePair] = useState<{ v1: string; v2: string } | null>(null);
  const [compareResult, setCompareResult] = useState<Record<string, any> | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const loadVersions = useCallback(async () => {
    if (!ontologyUri) return;
    setError("");
    try {
      const response = await fetch(`/api/ontology/versions/${encodeURIComponent(ontologyUri)}`);
      if (response.ok) {
        const data = await response.json();
        setVersions(data);
        if (response.status === 207) setError(data.message || i18next.t('ontologyTools.versions.partialVersions'));
      } else {
        setError(i18next.t('ontologyTools.versions.loadVersionsFailed', { status: response.status }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : i18next.t('ontologyTools.versions.loadVersionsError'));
    }
  }, [ontologyUri]);

  const loadProposals = useCallback(async () => {
    setError("");
    try {
      const response = await fetch("/api/ontology/proposals");
      if (response.ok) {
        const data = await response.json();
        setProposals(data);
        if (response.status === 207) setError(data.message || i18next.t('ontologyTools.versions.partialProposals'));
      } else {
        setError(i18next.t('ontologyTools.versions.loadProposalsFailed', { status: response.status }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : i18next.t('ontologyTools.versions.loadProposalsError'));
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    async function fetchInitial() {
      if (!ignore) setError("");
      try {
        const propRes = await fetch("/api/ontology/proposals");
        if (propRes.ok) {
          const propData = await propRes.json();
          if (!ignore) {
            setProposals(propData);
            if (propRes.status === 207) setError(propData.message || i18next.t('ontologyTools.versions.partialProposals'));
          }
        } else if (!ignore) {
          setError(i18next.t('ontologyTools.versions.loadProposalsFailed', { status: propRes.status }));
        }
      } catch (err) {
        if (!ignore) setError(err instanceof Error ? err.message : i18next.t('ontologyTools.versions.loadProposalsError'));
      }

      if (!ontologyUri) return;
      try {
        const verRes = await fetch(`/api/ontology/versions/${encodeURIComponent(ontologyUri)}`);
        if (verRes.ok) {
          const verData = await verRes.json();
          if (!ignore) {
            setVersions(verData);
            if (verRes.status === 207) setError(verData.message || i18next.t('ontologyTools.versions.partialVersions'));
          }
        } else if (!ignore) {
          setError((prev) => prev || i18next.t('ontologyTools.versions.loadVersionsFailed', { status: verRes.status }));
        }
      } catch (err) {
        if (!ignore) setError((prev) => prev || (err instanceof Error ? err.message : i18next.t('ontologyTools.versions.loadVersionsError')));
      }
    }
    void fetchInitial();
    return () => { ignore = true; };
  }, [ontologyUri]);

  const approveProposal = useCallback(async (proposalId: string) => {
    setError("");
    try {
      const response = await fetch(`/api/ontology/proposals/${proposalId}/approve`, {
        method: "POST",
      });
      if (response.ok) {
        if (response.status === 207) {
          const data = await response.json().catch(() => ({}));
          setError(data.message || i18next.t('ontologyTools.versions.partialApprove'));
        } else {
          alert(i18next.t('ontologyTools.common.proposalApproved'));
        }
        loadProposals();
      } else {
        setError(i18next.t('ontologyTools.versions.approveFailed', { status: response.status }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : i18next.t('ontologyTools.versions.approveError'));
    }
  }, [loadProposals]);

  const rejectProposal = useCallback(async (proposalId: string) => {
    setError("");
    try {
      const response = await fetch(`/api/ontology/proposals/${proposalId}/reject`, {
        method: "POST",
      });
      if (response.ok) {
        if (response.status === 207) {
          const data = await response.json().catch(() => ({}));
          setError(data.message || i18next.t('ontologyTools.versions.partialReject'));
        } else {
          alert(i18next.t('ontologyTools.common.proposalRejected'));
        }
        loadProposals();
      } else {
        setError(i18next.t('ontologyTools.versions.rejectFailed', { status: response.status }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : i18next.t('ontologyTools.versions.rejectError'));
    }
  }, [loadProposals]);

  const publishProposal = useCallback(async (proposalId: string) => {
    setError("");
    try {
      const response = await fetch(`/api/ontology/proposals/${proposalId}/publish`, {
        method: "POST",
      });
      if (response.ok) {
        if (response.status === 207) {
          const data = await response.json().catch(() => ({}));
          setError(data.message || i18next.t('ontologyTools.versions.partialPublish'));
        } else {
          alert(i18next.t('ontologyTools.common.proposalPublished'));
        }
        loadProposals();
        loadVersions();
      } else {
        setError(i18next.t('ontologyTools.versions.publishFailed', { status: response.status }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : i18next.t('ontologyTools.versions.publishError'));
    }
  }, [loadProposals, loadVersions]);

  const runVersionComparison = useCallback(async () => {
    if (!comparePair || !ontologyUri) return;
    setIsLoading(true);
    setError("");
    try {
      const response = await fetch(`/api/ontology/versions/${encodeURIComponent(ontologyUri)}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          version1: comparePair.v1,
          version2: comparePair.v2,
        }),
      });
      if (response.ok) {
        const data = await response.json();
        if (response.status === 207) setError(data.message || i18next.t('ontologyTools.versions.partialCompare'));
        setCompareResult(data);
      } else {
        setError(i18next.t('ontologyTools.versions.compareFailed', { status: response.status }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : i18next.t('ontologyTools.versions.compareError'));
    } finally {
      setIsLoading(false);
    }
  }, [comparePair, ontologyUri]);

  const getStateIcon = (state: string) => {
    switch (state) {
      case "published":
        return <CheckCircle size={16} color="#4cc38a" />;
      case "approved":
        return <CheckCircle size={16} color="#4aa3ff" />;
      case "rejected":
        return <XCircle size={16} color="#ff6b6b" />;
      case "proposed":
        return <AlertCircle size={16} color="#f2b66d" />;
      default:
        return <Clock size={16} color="#8fa8c6" />;
    }
  };

  const containerStyle: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    background: "#07111f",
    padding: "20px",
    overflow: "auto",
  };

  const headerStyle: React.CSSProperties = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "20px",
  };

  const titleStyle: React.CSSProperties = {
    margin: 0,
    color: "#ebf3ff",
    fontSize: "20px",
    fontWeight: "700",
  };

  const sectionStyle: React.CSSProperties = {
    marginBottom: "24px",
  };

  const sectionTitleStyle: React.CSSProperties = {
    margin: "0 0 12px",
    color: "#ebf3ff",
    fontSize: "14px",
    fontWeight: "600",
    display: "flex",
    alignItems: "center",
    gap: "8px",
  };

  const listStyle: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  };

  const itemStyle: React.CSSProperties = {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    padding: "12px 16px",
    borderRadius: "8px",
    background: "rgba(9, 19, 34, 0.8)",
    border: "1px solid rgba(127, 208, 255, 0.12)",
    transition: "160ms ease",
  };

  const modalOverlayStyle: React.CSSProperties = {
    position: "fixed",
    inset: 0,
    background: "rgba(0, 0, 0, 0.7)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
  };

  const modalStyle: React.CSSProperties = {
    background: "rgba(9, 19, 34, 0.95)",
    border: "1px solid rgba(127, 208, 255, 0.2)",
    borderRadius: "12px",
    padding: "24px",
    minWidth: "480px",
    maxWidth: "640px",
    maxHeight: "80vh",
    overflow: "auto",
    backdropFilter: "blur(18px)",
  };

  const buttonStyle: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    padding: "8px 14px",
    borderRadius: "6px",
    border: "1px solid rgba(127, 208, 255, 0.2)",
    background: "rgba(74, 163, 255, 0.1)",
    color: "#ebf3ff",
    fontSize: "12px",
    fontWeight: "600",
    cursor: "pointer",
    transition: "160ms ease",
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "10px 12px",
    borderRadius: "6px",
    border: "1px solid rgba(127, 208, 255, 0.2)",
    background: "rgba(3, 9, 18, 0.8)",
    color: "#ebf3ff",
    fontSize: "13px",
    marginBottom: "12px",
  };

  const errorStyle: React.CSSProperties = { padding: "12px", borderRadius: "14px", color: "#ffb4c2", background: "rgba(255,157,175,0.1)", border: "1px solid rgba(255,157,175,0.18)", marginBottom: "16px" };

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>
        <h1 style={titleStyle}>{t('ontologyTools.versions.title')}</h1>
        <input
          type="text"
          placeholder={t('ontologyTools.versions.ontologyUriPlaceholder')}
          value={ontologyUri}
          onChange={(e) => setOntologyUri(e.target.value)}
          style={{ ...inputStyle, width: "300px", marginBottom: 0 }}
        />
      </div>

      {error ? <div style={errorStyle}>{error}</div> : null}

      <div style={sectionStyle}>
        <h2 style={sectionTitleStyle}>
          <Layers size={16} />
          {t('ontologyTools.versions.versionHistory')}
        </h2>
        <div style={listStyle}>
          {versions.length === 0 ? (
            <div style={{ color: "#8fa8c6", fontSize: "13px" }}>{t('ontologyTools.versions.noVersions')}</div>
          ) : (
            versions.map((version) => (
              <div key={version.version_id} style={itemStyle}>
                {getStateIcon(version.state)}
                <div style={{ flex: 1 }}>
                  <div style={{ color: "#ebf3ff", fontSize: "13px", fontWeight: "600" }}>
                    {version.version_id}
                  </div>
                  <div style={{ color: "#8fa8c6", fontSize: "11px" }}>
                    {version.author} • {new Date(version.date).toLocaleDateString()}
                  </div>
                </div>
                <button
                  style={buttonStyle}
                  onClick={() => {
                    setComparePair({ v1: version.version_id, v2: versions[0]?.version_id || "" });
                    setShowCompareModal(true);
                  }}
                >
                  <ArrowRight size={12} />
                  {t('ontologyTools.compare')}
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div style={sectionStyle}>
        <h2 style={sectionTitleStyle}>
          <GitMerge size={16} />
          {t('ontologyTools.versions.changeProposals')}
        </h2>
        <div style={listStyle}>
          {proposals.length === 0 ? (
            <div style={{ color: "#8fa8c6", fontSize: "13px" }}>{t('ontologyTools.versions.noProposals')}</div>
          ) : (
            proposals.map((proposal) => (
              <div key={proposal.proposal_id} style={itemStyle}>
                {getStateIcon(proposal.state)}
                <div style={{ flex: 1 }}>
                  <div style={{ color: "#ebf3ff", fontSize: "13px", fontWeight: "600" }}>
                    {proposal.summary}
                  </div>
                  <div style={{ color: "#8fa8c6", fontSize: "11px" }}>
                    {proposal.author} • {new Date(proposal.created_at).toLocaleDateString()}
                  </div>
                </div>
                {proposal.state === "proposed" && (
                  <div style={{ display: "flex", gap: "6px" }}>
                    <button style={buttonStyle} onClick={() => approveProposal(proposal.proposal_id)}>
                      <CheckCircle size={12} />
                      {t('ontologyTools.approve')}
                    </button>
                    <button style={buttonStyle} onClick={() => rejectProposal(proposal.proposal_id)}>
                      <XCircle size={12} />
                      {t('ontologyTools.reject')}
                    </button>
                  </div>
                )}
                {proposal.state === "approved" && (
                  <button style={buttonStyle} onClick={() => publishProposal(proposal.proposal_id)}>
                    <Send size={12} />
                    {t('ontologyTools.publish')}
                  </button>
                )}
                <button
                  style={buttonStyle}
                  onClick={() => {
                    setSelectedProposal(proposal);
                    setShowProposalModal(true);
                  }}
                >
                  <FileText size={12} />
                  {t('ontologyTools.details')}
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {showProposalModal && selectedProposal && (
        <div style={modalOverlayStyle} onClick={() => setShowProposalModal(false)}>
          <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "16px" }}>
              <h3 style={{ margin: 0, color: "#ebf3ff", fontSize: "16px" }}>{t('ontologyTools.versions.proposalDetails')}</h3>
              <button onClick={() => setShowProposalModal(false)} style={{ background: "none", border: "none", color: "#8fa8c6", cursor: "pointer" }}>
                <X size={18} />
              </button>
            </div>
            <div style={{ marginBottom: "12px" }}>
              <label style={{ display: "block", color: "#8fa8c6", fontSize: "12px", marginBottom: "4px" }}>
                {t('ontologyTools.common.summary')}
              </label>
              <div style={{ color: "#ebf3ff", fontSize: "13px" }}>{selectedProposal.summary}</div>
            </div>
            <div style={{ marginBottom: "12px" }}>
              <label style={{ display: "block", color: "#8fa8c6", fontSize: "12px", marginBottom: "4px" }}>
                {t('ontologyTools.common.state')}
              </label>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "#ebf3ff", fontSize: "13px" }}>
                {getStateIcon(selectedProposal.state)}
                {t(`ontologyTools.states.${selectedProposal.state}`)}
              </div>
            </div>
            <div style={{ marginBottom: "12px" }}>
              <label style={{ display: "block", color: "#8fa8c6", fontSize: "12px", marginBottom: "4px" }}>
                {t('ontologyTools.common.impactAnalysis')}
              </label>
              <pre style={{ background: "rgba(3, 9, 18, 0.8)", padding: "12px", borderRadius: "6px", color: "#ebf3ff", fontSize: "12px", overflow: "auto" }}>
                {JSON.stringify(selectedProposal.impact_analysis, null, 2)}
              </pre>
            </div>
            <div style={{ marginBottom: "12px" }}>
              <label style={{ display: "block", color: "#8fa8c6", fontSize: "12px", marginBottom: "4px" }}>
                {t('ontologyTools.common.shaclValidation')}
              </label>
              <pre style={{ background: "rgba(3, 9, 18, 0.8)", padding: "12px", borderRadius: "6px", color: "#ebf3ff", fontSize: "12px", overflow: "auto" }}>
                {JSON.stringify(selectedProposal.shacl_validation, null, 2)}
              </pre>
            </div>
            <div>
              <label style={{ display: "block", color: "#8fa8c6", fontSize: "12px", marginBottom: "4px" }}>
                {t('ontologyTools.common.comments', { count: selectedProposal.comments.length })}
              </label>
              <div style={{ maxHeight: "120px", overflow: "auto" }}>
                {selectedProposal.comments.map((comment) => (
                  <div key={comment.id} style={{ padding: "8px", background: "rgba(3, 9, 18, 0.6)", borderRadius: "4px", marginBottom: "6px" }}>
                    <div style={{ color: "#ebf3ff", fontSize: "12px", fontWeight: "600" }}>{comment.author}</div>
                    <div style={{ color: "#8fa8c6", fontSize: "11px" }}>{comment.text}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {showCompareModal && comparePair && (
        <div style={modalOverlayStyle} onClick={() => setShowCompareModal(false)}>
          <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "16px" }}>
              <h3 style={{ margin: 0, color: "#ebf3ff", fontSize: "16px" }}>{t('ontologyTools.versions.compareVersions')}</h3>
              <button onClick={() => setShowCompareModal(false)} style={{ background: "none", border: "none", color: "#8fa8c6", cursor: "pointer" }}>
                <X size={18} />
              </button>
            </div>
            <div style={{ display: "flex", gap: "12px", marginBottom: "16px" }}>
              <div style={{ flex: 1 }}>
                <label style={{ display: "block", color: "#8fa8c6", fontSize: "12px", marginBottom: "4px" }}>
                  {t('ontologyTools.versions.version1')}
                </label>
                <input
                  type="text"
                  value={comparePair.v1}
                  onChange={(e) => setComparePair({ ...comparePair, v1: e.target.value })}
                  style={inputStyle}
                />
              </div>
              <div style={{ flex: 1 }}>
                <label style={{ display: "block", color: "#8fa8c6", fontSize: "12px", marginBottom: "4px" }}>
                  {t('ontologyTools.versions.version2')}
                </label>
                <input
                  type="text"
                  value={comparePair.v2}
                  onChange={(e) => setComparePair({ ...comparePair, v2: e.target.value })}
                  style={inputStyle}
                />
              </div>
            </div>
            <button style={buttonStyle} onClick={runVersionComparison} disabled={isLoading}>
              <Scale size={12} />
              {isLoading ? t('ontologyTools.comparing') : t('ontologyTools.compare')}
            </button>
            {compareResult && (
              <pre style={{ marginTop: "16px", background: "rgba(3, 9, 18, 0.8)", padding: "12px", borderRadius: "6px", color: "#ebf3ff", fontSize: "12px", overflow: "auto" }}>
                {JSON.stringify(compareResult, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
