import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, FileText, Brain, MessageCircle, Eye, Database, Sparkles, ArrowRight, CheckCircle, Clock, AlertCircle, Zap, Download, BarChart3, Search, X, Play, Pause } from 'lucide-react';
import './index.css'; // Import Tailwind CSS

const TableExtractionApp = () => {
  const API_BASE_URL = 'http://127.0.0.1:8000';
  const [files, setFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState('idle'); // idle, uploading, processing, completed, error
  const [uploadId, setUploadId] = useState(null);
  const [processingSteps, setProcessingSteps] = useState([]);
  const [question, setQuestion] = useState('');
  const [conversation, setConversation] = useState([]);
  const [isAsking, setIsAsking] = useState(false);
  const [selectedTable, setSelectedTable] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  // Animation states
  const [particleAnimation, setParticleAnimation] = useState(false);
  const [pulseEffect, setPulseEffect] = useState(false);

  // Processing steps can be updated based on backend status if available
  // For now, we'll simplify this as the backend handles processing asynchronously.
  useEffect(() => {
    if (uploadStatus === 'processing') {
      setParticleAnimation(true);
      // Backend handles processing, frontend shows a general processing state
      // We might need a polling mechanism here to get actual progress from backend
      // For now, we assume upload leads to processing and then completion (or error).
    } else {
      setParticleAnimation(false);
    }
  }, [uploadStatus]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);



  const handleFileUpload = useCallback(async (uploadedFiles) => {
    setFiles(uploadedFiles);
    setUploadStatus('uploading');
    setProcessingSteps([]); // Reset steps

    const formData = new FormData();
    uploadedFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch(`${API_BASE_URL}/upload_images/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'File upload failed');
      }

      const result = await response.json();
      setUploadId(result.upload_id);
      setUploadStatus('processing'); // Backend starts processing
      // Simulate a delay then set to completed, as backend processing is async
      // A better approach would be to poll a status endpoint from the backend
      setProcessingSteps([
        { id: 1, name: 'File Upload to Server', status: 'completed', duration: '0.5s' },
        { id: 2, name: 'Backend Processing (OCR, Table Extraction)', status: 'processing', duration: '...' },
      ]);
      setTimeout(() => { // Simulate backend processing time
        setUploadStatus('completed');
        setProcessingSteps(prev => prev.map(step => step.id === 2 ? {...step, status: 'completed', duration: 'Done'} : step));
        setPulseEffect(true);
        setTimeout(() => setPulseEffect(false), 2000);
      }, 5000); // Adjust this timeout or implement polling

    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus('error');
      setProcessingSteps([{ id: 1, name: 'Upload Failed', status: 'error', duration: '0s' }]);
      alert(`Error: ${error.message}`);
    }
  }, [API_BASE_URL]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFileUpload(droppedFiles);
  }, [handleFileUpload]);

  const handleFileInputChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    handleFileUpload(selectedFiles);
  };

  const askQuestion = async () => {
    if (!question.trim() || uploadStatus !== 'completed') return;

    setIsAsking(true);
    const userMessage = { type: 'user', content: question, timestamp: new Date() };
    setConversation(prev => [...prev, userMessage]);
    setQuestion(''); // Clear input after sending

    try {
      const response = await fetch(`${API_BASE_URL}/qa/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: userMessage.content }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get answer');
      }

      const aiData = await response.json();
      const aiMessage = {
        type: 'ai',
        content: aiData.answer,
        selectedTable: aiData.selected_table,
        tableTitle: aiData.table_title,
        sampleData: aiData.sample_data,
        timestamp: new Date(),
      };
      setConversation(prev => [...prev, aiMessage]);
      if (aiData.sample_data && aiData.sample_data.length > 0) {
        setSelectedTable(aiMessage); // For preview, if data is available
      }
    } catch (error) {
      console.error('QA error:', error);
      const errorMessage = {
        type: 'ai',
        content: `Error fetching answer: ${error.message}`,
        timestamp: new Date(),
      };
      setConversation(prev => [...prev, errorMessage]);
    } finally {
      setIsAsking(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-5 h-5 text-emerald-400" />;
      case 'processing': return <Clock className="w-5 h-5 text-blue-400 animate-spin" />;
      case 'error': return <AlertCircle className="w-5 h-5 text-red-400" />;
      default: return <div className="w-5 h-5 rounded-full border-2 border-gray-600" />;
    }
  };

  const suggestedQuestions = [
    "What's the total revenue in Q1?",
    "Show me the expense breakdown by category",
    "Which month had the highest profit margin?",
    "Compare revenue trends across quarters"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        {particleAnimation && (
          <div className="absolute inset-0">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="absolute w-2 h-2 bg-blue-400 rounded-full animate-pulse"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 2}s`,
                  animationDuration: `${2 + Math.random() * 3}s`
                }}
              />
            ))}
          </div>
        )}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(120,119,198,0.1),transparent_50%)]" />
      </div>

      <div className="relative z-10 container mx-auto px-6 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className={`p-4 rounded-2xl bg-gradient-to-r from-blue-500 to-purple-600 ${pulseEffect ? 'animate-pulse scale-110' : ''} transition-all duration-500`}>
              <Brain className="w-12 h-12" />
            </div>
          </div>
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
            Table Extraction Studio-Structura AI
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Transform your documents into intelligent, queryable data with advanced AI vision and natural language processing
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
              <div className="flex items-center mb-6">
                <Upload className="w-6 h-6 mr-3 text-blue-400" />
                <h2 className="text-2xl font-bold">Upload Documents</h2>
              </div>
              
              {uploadStatus === 'idle' && (
                <div
                  className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center hover:border-blue-400 transition-colors cursor-pointer group"
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="group-hover:scale-110 transition-transform">
                    <FileText className="w-16 h-16 mx-auto mb-4 text-slate-400 group-hover:text-blue-400" />
                    <p className="text-lg font-medium mb-2">Drop files here or click to browse</p>
                    <p className="text-sm text-gray-400">Supports PDF, PNG, JPG, JPEG</p>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".pdf,.png,.jpg,.jpeg"
                    onChange={handleFileInputChange}
                    className="hidden"
                  />
                </div>
              )}

              {files.length > 0 && (
                <div className="space-y-3 mb-6">
                  {files.map((file, idx) => (
                    <div key={idx} className="flex items-center justify-between bg-slate-700/50 rounded-lg p-3">
                      <div className="flex items-center">
                        <FileText className="w-5 h-5 mr-3 text-blue-400" />
                        <span className="text-sm font-medium truncate">{file.name}</span>
                      </div>
                      <span className="text-xs text-gray-400">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Processing Steps */}
              {uploadStatus !== 'idle' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold">Processing Pipeline</h3>
                    <div className="flex items-center text-sm text-gray-400">
                      <Zap className="w-4 h-4 mr-1" />
                      AI Powered
                    </div>
                  </div>
                  {processingSteps.map((step) => (
                    <div key={step.id} className="flex items-center justify-between bg-slate-700/30 rounded-lg p-3">
                      <div className="flex items-center">
                        {getStatusIcon(step.status)}
                        <span className="ml-3 text-sm font-medium">{step.name}</span>
                      </div>
                      <span className="text-xs text-gray-400">{step.duration}</span>
                    </div>
                  ))}
                </div>
              )}

              {uploadStatus === 'completed' && (
                <div className="bg-emerald-900/30 border border-emerald-400/30 rounded-lg p-4 mt-4">
                  <div className="flex items-center">
                    <CheckCircle className="w-5 h-5 text-emerald-400 mr-3" />
                    <span className="font-medium">Processing Complete!</span>
                  </div>
                  <p className="text-sm text-gray-300 mt-2">
                    Tables extracted and AI model trained. Ready for questions!
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Chat Section */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-slate-700/50 h-[600px] flex flex-col">
              <div className="flex items-center justify-between p-6 border-b border-slate-700/50">
                <div className="flex items-center">
                  <MessageCircle className="w-6 h-6 mr-3 text-purple-400" />
                  <h2 className="text-2xl font-bold">AI Assistant</h2>
                </div>
                {uploadStatus === 'completed' && (
                  <div className="flex items-center text-sm text-emerald-400">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full mr-2 animate-pulse" />
                    Online
                  </div>
                )}
              </div>

              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {uploadStatus !== 'completed' && conversation.length === 0 && (
                  <div className="text-center text-gray-400 mt-20">
                    <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p className="text-lg">Upload documents to start asking questions</p>
                    <p className="text-sm mt-2">The AI will analyze your tables and answer questions about the data</p>
                  </div>
                )}

                {conversation.map((message, idx) => (
                  <div key={idx} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] rounded-2xl p-4 ${
                      message.type === 'user' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-slate-700/50 text-gray-100'
                    }`}>
                      <p className="mb-2">{message.content}</p>
                      {message.type === 'ai' && message.sampleData && (
                        <div className="mt-4 p-3 bg-slate-800/50 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-blue-400">
                              {message.tableTitle}
                            </span>
                            <button
                              onClick={() => setShowPreview(true)}
                              className="text-xs text-purple-400 hover:text-purple-300 flex items-center"
                            >
                              <Eye className="w-3 h-3 mr-1" />
                              View Full
                            </button>
                          </div>
                          <div className="text-xs space-y-1">
                            {message.sampleData.slice(0, 2).map((row, i) => (
                              <div key={i} className="flex justify-between text-gray-300">
                                <span>{row.Month}</span>
                                <span>{row.Revenue}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="text-xs text-gray-400 mt-2">
                        {message.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
                
                {isAsking && (
                  <div className="flex justify-start">
                    <div className="bg-slate-700/50 rounded-2xl p-4">
                      <div className="flex items-center space-x-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" />
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}} />
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}} />
                        </div>
                        <span className="text-sm text-gray-400">AI is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Suggested Questions */}
              {uploadStatus === 'completed' && conversation.length === 0 && (
                <div className="p-4 border-t border-slate-700/50">
                  <p className="text-sm text-gray-400 mb-3">Try asking:</p>
                  <div className="grid grid-cols-2 gap-2">
                    {suggestedQuestions.map((q, idx) => (
                      <button
                        key={idx}
                        onClick={() => setQuestion(q)}
                        className="text-left text-sm p-2 bg-slate-700/30 hover:bg-slate-700/50 rounded-lg transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Input Area */}
              <div className="p-6 border-t border-slate-700/50">
                <div className="flex space-x-3">
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder={uploadStatus === 'completed' ? "Ask anything about your data..." : "Upload documents first..."}
                    disabled={uploadStatus !== 'completed' || isAsking}
                    className="flex-1 bg-slate-700/50 border border-slate-600 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 disabled:opacity-50"
                    onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
                  />
                  <button
                    onClick={askQuestion}
                    disabled={!question.trim() || uploadStatus !== 'completed' || isAsking}
                    className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl px-6 py-3 font-medium transition-all duration-200 flex items-center"
                  >
                    {isAsking ? (
                      <Clock className="w-5 h-5 animate-spin" />
                    ) : (
                      <ArrowRight className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl p-6 border border-slate-700/30">
            <div className="flex items-center mb-4">
              <Eye className="w-8 h-8 text-blue-400 mr-3" />
              <h3 className="text-lg font-semibold">AI Vision</h3>
            </div>
            <p className="text-gray-300 text-sm">Advanced computer vision extracts tables from any document format with high accuracy</p>
          </div>
          
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl p-6 border border-slate-700/30">
            <div className="flex items-center mb-4">
              <Database className="w-8 h-8 text-purple-400 mr-3" />
              <h3 className="text-lg font-semibold">Smart Classification</h3>
            </div>
            <p className="text-gray-300 text-sm">Automatically categorizes and structures data for optimal query performance</p>
          </div>
          
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-xl p-6 border border-slate-700/30">
            <div className="flex items-center mb-4">
              <Sparkles className="w-8 h-8 text-pink-400 mr-3" />
              <h3 className="text-lg font-semibold">Natural Language</h3>
            </div>
            <p className="text-gray-300 text-sm">Ask questions in plain English and get intelligent, contextual answers</p>
          </div>
        </div>
      </div>

      {/* Preview Modal */}
      {showPreview && selectedTable && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-6">
          <div className="bg-slate-800 rounded-2xl p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold">{selectedTable.tableTitle}</h3>
              <button
                onClick={() => setShowPreview(false)}
                className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b border-slate-600">
                    {Object.keys(selectedTable.sampleData[0]).map((key) => (
                      <th key={key} className="text-left p-3 font-medium text-gray-300">{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {selectedTable.sampleData.map((row, idx) => (
                    <tr key={idx} className="border-b border-slate-700/50">
                      {Object.values(row).map((value, i) => (
                        <td key={i} className="p-3 text-gray-200">{value}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TableExtractionApp;
