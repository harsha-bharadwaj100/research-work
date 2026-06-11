import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { UploadCloud, ShieldCheck, AlertTriangle, Loader2, Zap, Eye, Binary, ShieldAlert, FileSearch, Download } from 'lucide-react';
import { jsPDF } from "jspdf";
import html2canvas from "html2canvas";
import './index.css';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const reportRef = React.useRef(null);


  const [activeMode, setActiveMode] = useState('single');
  const [imageA, setImageA] = useState(null);
  const [previewA, setPreviewA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [previewB, setPreviewB] = useState(null);
  const [dualResult, setDualResult] = useState(null);

  const onDropA = useCallback((acceptedFiles) => {
    setImageA(acceptedFiles[0]);
    setPreviewA(URL.createObjectURL(acceptedFiles[0]));
    setDualResult(null);
  }, []);

  const onDropB = useCallback((acceptedFiles) => {
    setImageB(acceptedFiles[0]);
    setPreviewB(URL.createObjectURL(acceptedFiles[0]));
    setDualResult(null);
  }, []);

  const { getRootProps: getRootPropsA, getInputProps: getInputPropsA } = useDropzone({ onDrop: onDropA, accept: 'image/*' });
  const { getRootProps: getRootPropsB, getInputProps: getInputPropsB } = useDropzone({ onDrop: onDropB, accept: 'image/*' });

  const runDualAnalysis = async () => {
    if (!imageA || !imageB) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('image1', imageA);
    formData.append('image2', imageB);

    try {
      const response = await axios.post('http://localhost:8000/compare', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setDualResult(response.data);
    } catch (error) {
      console.error("Error", error);
      alert("Failed to connect to backend");
    }
    setLoading(false);
  };

  const onDrop = useCallback((acceptedFiles) => {
    setImage(acceptedFiles[0]);
    setPreview(URL.createObjectURL(acceptedFiles[0]));
    setResult(null);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({ onDrop, accept: 'image/*' });

  const runAnalysis = async () => {
    if (!image) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (error) {
      console.error("Error running analysis", error);
      alert("Failed to connect to GlassEngine Backend. Make sure it's running on port 8000.");
    }
    setLoading(false);
  };

  const downloadPDF = async () => {
    if (!reportRef.current) return;
    
    // Create professional jsPDF document
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pageWidth = pdf.internal.pageSize.getWidth();
    const margin = 15;
    let currentY = 20;

    // 1. Header & Branding
    pdf.setFillColor(10, 10, 10); // Dark background for header
    pdf.rect(0, 0, pageWidth, 40, 'F');
    pdf.setTextColor(0, 255, 255); // Neon Cyan
    pdf.setFontSize(24);
    pdf.text("PROJECT GLASSENGINE", margin, 25);
    pdf.setFontSize(10);
    pdf.setTextColor(255, 255, 255);
    pdf.text("OFFICIAL FORENSIC VERIFICATION AUDIT", margin, 32);
    pdf.text(`Date: ${new Date().toLocaleString()}`, pageWidth - margin - 50, 32);
    
    currentY = 55;

    // 2. Executive Summary Section
    pdf.setTextColor(0, 0, 0);
    pdf.setFontSize(16);
    pdf.setFont("helvetica", "bold");
    pdf.text("EXECUTIVE SUMMARY", margin, currentY);
    currentY += 10;
    
    pdf.setFontSize(12);
    pdf.setFont("helvetica", "normal");
    const verdictColor = result.verdict.includes('Authentic') ? [0, 150, 0] : [200, 0, 0];
    pdf.setTextColor(verdictColor[0], verdictColor[1], verdictColor[2]);
    pdf.text(`PRIMARY VERDICT: ${result.verdict.toUpperCase()}`, margin, currentY);
    currentY += 7;
    
    pdf.setTextColor(80, 80, 80);
    const splitExplanation = pdf.splitTextToSize(result.explanation, pageWidth - (margin * 2));
    pdf.text(splitExplanation, margin, currentY);
    currentY += (splitExplanation.length * 6) + 10;

    // 3. Forensic Integrity Matrix (Manual Table)
    pdf.setTextColor(0, 0, 0);
    pdf.setFontSize(14);
    pdf.setFont("helvetica", "bold");
    pdf.text("FORENSIC INTEGRITY MATRIX", margin, currentY);
    currentY += 8;

    const tableHeader = ["Metric", "Range", "Reading", "Deviation"];
    const colWidths = [50, 40, 40, 50];
    
    // Header Row
    pdf.setFillColor(240, 240, 240);
    pdf.rect(margin, currentY, pageWidth - (margin * 2), 10, 'F');
    pdf.setFontSize(10);
    let tempX = margin + 2;
    tableHeader.forEach((h, i) => {
      pdf.text(h, tempX, currentY + 7);
      tempX += colWidths[i];
    });
    currentY += 10;

    // Data Rows
    const data = [
      ["Inter-Channel Phase", "450 - 5000+", result.ratio.toFixed(2), result.ratio < 450 ? "LOW" : "STABLE"],
      ["Shannon Entropy", "7.2 - 7.8", result.entropy.toFixed(3), (result.entropy < 7.2 || result.entropy > 7.8) ? "ABNORMAL" : "NATURAL"],
      ["Sensor Noise (Var)", "100 - 220", result.noise_variance.toFixed(2), (result.noise_variance < 100 || result.noise_variance > 230) ? "HIGH" : "UNIFORM"],
      ["Compression (ELA)", "15 - 22", result.ela_score.toFixed(2), (result.ela_score < 14 || result.ela_score > 23) ? "SUSPICIOUS" : "HEALTHY"]
    ];

    pdf.setFont("helvetica", "normal");
    data.forEach(row => {
      tempX = margin + 2;
      row.forEach((cell, i) => {
        if (i === 3 && (cell === "LOW" || cell === "ABNORMAL" || cell === "HIGH" || cell === "SUSPICIOUS")) {
          pdf.setTextColor(200, 0, 0);
        } else {
          pdf.setTextColor(0, 0, 0);
        }
        pdf.text(cell.toString(), tempX, currentY + 7);
        tempX += colWidths[i];
      });
      pdf.line(margin, currentY + 10, pageWidth - margin, currentY + 10);
      currentY += 10;
    });

    currentY += 15;

    // 4. Visual Evidence Captures
    pdf.setTextColor(0, 0, 0);
    pdf.setFontSize(14);
    pdf.setFont("helvetica", "bold");
    pdf.text("DETAILED FORENSIC GRAPHICS", margin, currentY);
    currentY += 10;

    const visualElements = document.querySelectorAll('.visual-item');
    for (let i = 0; i < visualElements.length; i++) {
      if (currentY > 230) { pdf.addPage(); currentY = 20; }
      
      const canvas = await html2canvas(visualElements[i], { scale: 2, backgroundColor: '#ffffff' });
      const imgData = canvas.toDataURL('image/jpeg', 0.8);
      
      const imgWidth = 85; 
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      
      // Arrange 2x2 grid in PDF or sequential
      const xPos = (i % 2 === 0) ? margin : margin + 90;
      pdf.addImage(imgData, 'JPEG', xPos, currentY, imgWidth, imgHeight);
      
      if (i % 2 !== 0 || i === visualElements.length - 1) {
        currentY += imgHeight + 15;
      }
    }

    pdf.save(`GlassEngine_Audit_Report_${new Date().getTime()}.pdf`);
  };

  return (
    <>
      <div className="app-bg"></div>
      <div className="app-container">
        <header className="header">
          <h1>PROJECT GLASSENGINE</h1>
          <p>Single & Dual Image Analysis</p>
          <div style={{display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '20px'}}>
             <button onClick={() => setActiveMode('single')} style={{padding: '10px 20px', background: activeMode === 'single' ? 'var(--neon-blue)' : 'transparent', border: '1px solid var(--neon-blue)', color: 'white', borderRadius: '5px', cursor: 'pointer', transition: 'all 0.3s'}}>1. Single Neural Audit (Live)</button>
             <button onClick={() => setActiveMode('dual')} style={{padding: '10px 20px', background: activeMode === 'dual' ? 'var(--neon-green)' : 'transparent', border: '1px solid var(--neon-green)', color: 'white', borderRadius: '5px', cursor: 'pointer', transition: 'all 0.3s'}}>2. Dual Math Comparison (Original)</button>
          </div>
        </header>

        

        {activeMode === 'dual' ? (
           !dualResult && !loading ? (
             <main className="upload-section">
                <div style={{display: 'flex', gap: '40px', justifyContent: 'center', flexWrap: 'wrap'}}>
                  <div className="dropzone-wrapper" style={{width: '400px'}}>
                    <h3 style={{textAlign: 'center', marginBottom: '10px', color: 'var(--neon-blue)'}}>Suspect Image 1</h3>
                    <div {...getRootPropsA()} className={`dropzone ${previewA ? 'has-image' : ''}`} style={{ height: '300px' }}>
                      <input {...getInputPropsA()} />
                      {previewA ? <img src={previewA} className="preview-img" style={{ objectFit: 'contain' }} /> : <div className="dropzone-placeholder"><FileSearch size={48} /><p>Drop Image 1 Here</p></div>}
                    </div>
                  </div>
                  <div className="dropzone-wrapper" style={{width: '400px'}}>
                    <h3 style={{textAlign: 'center', marginBottom: '10px', color: 'var(--neon-blue)'}}>Suspect Image 2</h3>
                    <div {...getRootPropsB()} className={`dropzone ${previewB ? 'has-image' : ''}`} style={{ height: '300px' }}>
                      <input {...getInputPropsB()} />
                      {previewB ? <img src={previewB} className="preview-img" style={{ objectFit: 'contain' }} /> : <div className="dropzone-placeholder"><FileSearch size={48} /><p>Drop Image 2 Here</p></div>}
                    </div>
                  </div>
                </div>
                {previewA && previewB && (
                  <div className="btn-container" style={{marginTop: '30px'}}>
                    <button className="analyze-btn" style={{background: 'var(--neon-green)'}} onClick={runDualAnalysis}>
                       <Zap size={28} /> INITIATE PURE MATH COMPARISON
                    </button>
                  </div>
                )}
             </main>
           ) : loading ? (
             <main className="upload-section">
                <div className="btn-container" style={{ marginTop: '10vh' }}>
                  <button className="analyze-btn"><Loader2 className="spinner" size={36} /> CALCULATING PURE MATH METRICS...</button>
                </div>
             </main>
           ) : (
             <main className="results-section">
                <div className="verdict-banner" style={{ borderColor: 'var(--neon-green)' }}>
                  <h2 style={{color: 'var(--neon-green)'}}><ShieldCheck size={40} style={{verticalAlign: 'bottom'}} /> {dualResult.overall_verdict}</h2>
                </div>
                
                <div style={{display: 'flex', gap: '30px', marginTop: '40px', flexWrap: 'wrap', alignItems: 'flex-start'}}>
                  {[dualResult.image1, dualResult.image2].map((res, idx) => (
                    <div key={idx} style={{flex: 1, minWidth: '400px', background: 'rgba(255,255,255,0.03)', padding: '30px', borderRadius: '15px', border: `2px solid ${res.verdict.includes('Authentic') ? 'var(--neon-green)' : 'var(--neon-red)'}`}}>
                       <h3 style={{textAlign: 'center', fontSize: '1.5rem', color: res.verdict.includes('Authentic') ? 'var(--neon-green)' : 'var(--neon-red)'}}>
                          Image {idx+1}: {res.verdict}
                       </h3>
                       <p style={{fontSize: '0.9rem', opacity: 0.8, textAlign: 'center', marginTop: '10px'}}>{res.explanation}</p>
                       <img src={idx === 0 ? previewA : previewB} style={{width: '100%', height: '250px', objectFit: 'contain', margin: '20px 0', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.2)'}} />
                       
                       <div className="comparison-table-container" style={{ background: 'rgba(0,0,0,0.3)', padding: '15px', borderRadius: '10px', marginBottom: '20px' }}>
                         <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                           <tbody>
                             <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                               <td style={{ padding: '8px', opacity: 0.8 }}>ICPC PMR</td>
                               <td style={{ padding: '8px', fontWeight: 'bold' }}>{res.ratio.toFixed(2)}</td>
                             </tr>
                             <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                               <td style={{ padding: '8px', opacity: 0.8 }}>Rényi Entropy</td>
                               <td style={{ padding: '8px', fontWeight: 'bold' }}>{res.entropy.toFixed(3)}</td>
                             </tr>
                             <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                               <td style={{ padding: '8px', opacity: 0.8 }}>Noise Var.</td>
                               <td style={{ padding: '8px', fontWeight: 'bold' }}>{res.noise_variance.toFixed(2)}</td>
                             </tr>
                             <tr>
                               <td style={{ padding: '8px', opacity: 0.8 }}>ELA Score</td>
                               <td style={{ padding: '8px', fontWeight: 'bold' }}>{res.ela_score.toFixed(2)}</td>
                             </tr>
                           </tbody>
                         </table>
                       </div>

                       <div style={{marginTop: '20px'}}>
                         <p style={{fontSize: '0.85rem', color: 'var(--neon-purple)', marginBottom: '5px'}}><ShieldCheck size={14}/> HF-ICPC Sync</p>
                         <img src={res.icpc_graph} style={{width: '100%', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.1)'}} />
                       </div>
                       <div style={{marginTop: '20px'}}>
                         <p style={{fontSize: '0.85rem', color: 'var(--neon-purple)', marginBottom: '5px'}}><Eye size={14}/> Chaos Map</p>
                         <img src={res.entropy_map} style={{width: '100%', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.1)'}} />
                       </div>
                    </div>
                  ))}
                </div>
                <div className="btn-container" style={{marginTop: '40px'}}>
                   <button className="reset-btn" onClick={() => {setDualResult(null); setImageA(null); setImageB(null); setPreviewA(null); setPreviewB(null);}}>Audit New Pair</button>
                </div>
             </main>
           )
        ) : (

          !result && !loading ? (
          <main className="upload-section">
            <div className="upload-panels" style={{ flexDirection: 'column', alignItems: 'center' }}>
              <div className="dropzone-wrapper" style={{ width: '100%', maxWidth: '600px' }}>
                <div {...getRootProps()} className={`dropzone ${preview ? 'has-image' : ''}`} style={{ height: '400px' }}>
                  <input {...getInputProps()} />
                  {preview ? (
                    <img src={preview} alt="Preview" className="preview-img" style={{ objectFit: 'contain' }} />
                  ) : (
                    <div className="dropzone-placeholder">
                      <FileSearch size={72} style={{ marginBottom: '20px' }} />
                      <p style={{ fontSize: '1.2rem', margin: 0 }}>Drop Suspect Image Here</p>
                      <p style={{ fontSize: '0.9rem', opacity: 0.6 }}>Or click to browse files</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {preview && (
              <div className="btn-container">
                <button className="analyze-btn" onClick={runAnalysis}>
                  <Zap size={28} /> INITIATE SINGLE-IMAGE AUDIT
                </button>
              </div>
            )}
          </main>
        ) : loading ? (
          <main className="upload-section">
            <div className="btn-container" style={{ marginTop: '10vh' }}>
              <button className="analyze-btn">
                <Loader2 className="spinner" size={36} /> CALCULATING ENTROPY-PHASE RATIO...
              </button>
            </div>
          </main>
        ) : (
          <main className="results-section" ref={reportRef}>
            <div className="verdict-banner" style={{ borderColor: result.verdict.includes('Authentic') ? 'var(--neon-green)' : 'var(--neon-red)' }}>
              <h2 style={{ color: result.verdict.includes('Authentic') ? 'var(--neon-green)' : 'var(--neon-red)' }}>
                {result.verdict.includes('Authentic') ? <ShieldCheck size={40} style={{ verticalAlign: 'bottom' }} /> : <AlertTriangle size={40} style={{ verticalAlign: 'bottom' }} />}
                {' '} {result.verdict}
              </h2>
              <p className="explanation-text">{result.explanation}</p>

              <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '20px', marginTop: '20px' }}>
                <div className="metrics-box" style={{ background: 'rgba(255,255,255,0.05)', padding: '10px 20px', borderRadius: '8px', textAlign: 'center' }}>
                  <div className="metric-label" style={{ fontSize: '0.8rem', opacity: 0.6 }}>ICPC PMR</div>
                  <div className="metric-value" style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{result.ratio.toFixed(2)}</div>
                  <div className="metric-range" style={{ fontSize: '0.7rem', opacity: 0.5 }}>Typical: &gt; 600</div>
                </div>
                <div className="metrics-box" style={{ background: 'rgba(255,255,255,0.05)', padding: '10px 20px', borderRadius: '8px', textAlign: 'center' }}>
                  <div className="metric-label" style={{ fontSize: '0.8rem', opacity: 0.6 }}>Noise Variance</div>
                  <div className="metric-value" style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{result.noise_variance?.toFixed(2) || 'N/A'}</div>
                  <div className="metric-range" style={{ fontSize: '0.7rem', opacity: 0.5 }}>Typical: 100-220</div>
                </div>
                <div className="metrics-box" style={{ background: 'rgba(255,255,255,0.05)', padding: '10px 20px', borderRadius: '8px', textAlign: 'center' }}>
                  <div className="metric-label" style={{ fontSize: '0.8rem', opacity: 0.6 }}>ELA Score</div>
                  <div className="metric-value" style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>{result.ela_score?.toFixed(2) || 'N/A'}</div>
                  <div className="metric-range" style={{ fontSize: '0.7rem', opacity: 0.5 }}>Typical: 18-22</div>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '30px' }}>
                <button className="reset-btn" onClick={() => { setResult(null); setImage(null); setPreview(null); }}>
                  Audit Next Image
                </button>
                <button className="reset-btn" onClick={downloadPDF} style={{ background: 'var(--neon-blue)', color: 'white', borderColor: 'var(--neon-blue)' }}>
                  <Download size={18} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Download Forensic Report (PDF)
                </button>
              </div>
            </div>

            <div className="results-grid" style={{ gridTemplateColumns: '1fr', maxWidth: '800px', margin: '0 auto' }}>
              <div className={`result-card ${result.verdict.includes('Authentic') ? 'authentic' : 'fake'}`}>
                <div className="card-header">
                  <h3>Forensic Evidence</h3>
                </div>

                <div className="original-image">
                  <img src={preview} alt="Analyzed Subject" />
                </div>

                <div className="visuals" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', display: 'grid', gap: '20px', marginTop: '30px' }}>
                  <div className="visual-item" style={{ background: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '12px' }}>
                    <h4><ShieldCheck size={20} color="var(--neon-purple)" /> Inter-Channel Phase Correlation</h4>
                    <img src={result.icpc_graph} alt="ICPC Histogram" style={{ width: '100%', borderRadius: '8px', marginBottom: '10px' }} />
                    <p style={{ fontSize: '0.8rem', opacity: 0.7, lineHeight: 1.4 }}>
                      Measures the physical "Bayer Lock" between R, G, and B sensors. A sharp spike at 0 indicates a real camera; a flat distribution exposes AI generation.
                    </p>
                  </div>
                  <div className="visual-item" style={{ background: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '12px' }}>
                    <h4><Eye size={20} color="var(--neon-purple)" /> Shannon Entropy Heatmap</h4>
                    <img src={result.entropy_map} alt="Entropy Map" style={{ width: '100%', borderRadius: '8px', marginBottom: '10px' }} />
                    <p style={{ fontSize: '0.8rem', opacity: 0.7, lineHeight: 1.4 }}>
                      Detects local information density. AI-generated images often contain "zones of low complexity" where the math smoothed out the real-world micro-chaos.
                    </p>
                  </div>
                  <div className="visual-item" style={{ background: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '12px' }}>
                    <h4><Zap size={20} color="var(--neon-green)" /> Sensor Noise Residual</h4>
                    <img src={result.noise_map} alt="Noise Map" style={{ width: '100%', borderRadius: '8px', marginBottom: '10px' }} />
                    <p style={{ fontSize: '0.8rem', opacity: 0.7, lineHeight: 1.4 }}>
                      Exposes the "Digital Birthmark" (PRNU). Real sensors have microscopic hardware defects; AI images are "sterile" and lack this physical noise grain.
                    </p>
                  </div>
                  <div className="visual-item" style={{ background: 'rgba(255,255,255,0.03)', padding: '15px', borderRadius: '12px' }}>
                    <h4><ShieldAlert size={20} color="var(--neon-red)" /> Error Level Analysis (ELA)</h4>
                    <img src={result.ela_map} alt="ELA Map" style={{ width: '100%', borderRadius: '8px', marginBottom: '10px' }} />
                    <p style={{ fontSize: '0.8rem', opacity: 0.7, lineHeight: 1.4 }}>
                      Highlights JPEG recompression mismatches. Deepfakes often "glow" in ELA because the AI-generated pixels age differently than the rest of the image.
                    </p>
                  </div>
                </div>
              </div>

              {/* Forensic Integrity Matrix Table */}
              <div className="comparison-table-container" style={{ marginTop: '40px', background: 'rgba(255,255,255,0.02)', padding: '30px', borderRadius: '15px', border: '1px solid rgba(255,255,255,0.1)' }}>
                <h3 style={{ marginBottom: '20px', fontSize: '1.4rem', letterSpacing: '2px', color: 'var(--neon-blue)' }}>Forensic Integrity Matrix</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid rgba(255,255,255,0.1)' }}>
                      <th style={{ padding: '12px', opacity: 0.7 }}>Forensic Metric</th>
                      <th style={{ padding: '12px', opacity: 0.7 }}>Healthy Range (Camera)</th>
                      <th style={{ padding: '12px', opacity: 0.7 }}>Current Reading</th>
                      <th style={{ padding: '12px', opacity: 0.7 }}>Analytical Deviation</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '15px' }}>Inter-Channel Phase (PMR)</td>
                      <td style={{ padding: '15px' }}>450.0 - 5000.0+</td>
                      <td style={{ padding: '15px' }}>{result.ratio.toFixed(2)}</td>
                      <td style={{ padding: '15px', color: result.ratio < 450 ? 'var(--neon-red)' : 'var(--neon-green)' }}>
                        {result.ratio < 450 ? `Critically Low (${((400 / (result.ratio + 1)) * 100).toFixed(1)}% Weakness)` : 'Stable Bayer Lock'}
                      </td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '15px' }}>Shannon Chaos (Entropy)</td>
                      <td style={{ padding: '15px' }}>7.20 - 7.80</td>
                      <td style={{ padding: '15px' }}>{result.entropy.toFixed(3)}</td>
                      <td style={{ padding: '15px', color: (result.entropy < 7.2 || result.entropy > 7.8) ? 'var(--neon-red)' : 'var(--neon-green)' }}>
                        {(result.entropy < 7.2 || result.entropy > 7.8) ? 'Abnormal Smoothness' : 'Natural Entropy'}
                      </td>
                    </tr>
                    <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '15px' }}>Sensor Pattern Noise (Var)</td>
                      <td style={{ padding: '15px' }}>100.0 - 220.0</td>
                      <td style={{ padding: '15px' }}>{result.noise_variance.toFixed(2)}</td>
                      <td style={{ padding: '15px', color: (result.noise_variance < 100 || result.noise_variance > 230) ? 'var(--neon-red)' : 'var(--neon-green)' }}>
                        {(result.noise_variance < 100 || result.noise_variance > 230) ? `Abnormal (${((result.noise_variance / 160) * 100).toFixed(1)}% Artificial Noise)` : 'Hardware Uniformity'}
                      </td>
                    </tr>
                    <tr>
                      <td style={{ padding: '15px' }}>Compression History (ELA)</td>
                      <td style={{ padding: '15px' }}>15.0 - 22.0</td>
                      <td style={{ padding: '15px' }}>{result.ela_score.toFixed(2)}</td>
                      <td style={{ padding: '15px', color: (result.ela_score < 14 || result.ela_score > 23) ? 'var(--neon-red)' : 'var(--neon-green)' }}>
                        {(result.ela_score < 14 || result.ela_score > 23) ? 'Suspicious Processing Age' : 'Healthy Metadata Integrity'}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </main>
        )
        )}
      </div>
    </>
  );
}

export default App;
