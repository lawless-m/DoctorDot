import React, { useState } from 'react';
import { Citation } from '../types';

interface CitationDisplayProps {
  citations: Citation[];
}

const CitationDisplay: React.FC<CitationDisplayProps> = ({ citations }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-blue-700 hover:text-white hover:bg-blue-600 font-semibold text-sm transition-all px-4 py-2 rounded-xl bg-blue-50 border-2 border-blue-200 hover:border-blue-600 shadow-sm hover:shadow-md"
      >
        <span className="font-bold">{expanded ? '−' : '+'}</span>
        <span>{expanded ? 'Hide' : 'Show'} {citations.length} source{citations.length !== 1 ? 's' : ''}</span>
      </button>

      {expanded && (
        <div className="mt-3 space-y-2">
          {citations.map((citation, index) => (
            <div
              key={citation.chunk_id}
              className="bg-gradient-to-br from-blue-50 to-blue-100/50 border-2 border-blue-200 rounded-xl p-5 hover:border-blue-400 transition-all shadow-sm hover:shadow-md"
            >
              <div className="flex items-start gap-3">
                <div className="w-7 h-7 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex-shrink-0 flex items-center justify-center mt-0.5 shadow-sm">
                  <span className="text-white text-xs font-bold">{index + 1}</span>
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-gray-900 text-sm">
                    {citation.document_name.replace('.pdf', '')}
                    {citation.page_number && (
                      <span className="text-gray-600 font-normal"> • Page {citation.page_number}</span>
                    )}
                  </div>
                  <div className="text-gray-700 mt-2 text-xs leading-relaxed italic">
                    "{citation.chunk_text}"
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CitationDisplay;
