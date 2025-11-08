import React, { useState } from 'react';
import { Citation } from '../types';

interface CitationDisplayProps {
  citations: Citation[];
}

const CitationDisplay: React.FC<CitationDisplayProps> = ({ citations }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-2 text-sm">
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-blue-600 hover:text-blue-800 font-medium"
      >
        {expanded ? '▼' : '▶'} Sources ({citations.length})
      </button>

      {expanded && (
        <div className="mt-2 space-y-2">
          {citations.map((citation, index) => (
            <div
              key={citation.chunk_id}
              className="bg-gray-50 border border-gray-200 rounded p-3"
            >
              <div className="font-medium text-gray-700">
                {index + 1}. {citation.document_name}
                {citation.page_number && ` (Page ${citation.page_number})`}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                Relevance: {(citation.relevance_score * 100).toFixed(1)}%
              </div>
              <div className="text-gray-600 mt-2 text-xs italic">
                "{citation.chunk_text}"
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CitationDisplay;
