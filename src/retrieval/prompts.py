"""Prompts for query decomposition and classification."""

from __future__ import annotations

DECOMPOSITION_PROMPT = '''You are a Florida tax law expert assistant.

Given a tax question, break it down into specific sub-questions that need to be answered.

Consider these aspects:
1. Definitions: What terms need to be defined under Florida law?
2. Applicable statutes: Which sections of Florida Statutes (especially Chapter 212 for sales tax) apply?
3. Applicable rules: Which Department of Revenue rules (12A-1.xxx) implement those statutes?
4. Exemptions: Are there any exemptions that might apply?
5. Local taxes: Are there county-specific surtaxes (Miami-Dade, Hillsborough, etc.)?
6. Temporal: What time period or tax year is relevant?

IMPORTANT GUIDELINES:
- Only decompose if the query is genuinely complex (multiple aspects to consider)
- Simple queries like "What is the sales tax rate?" should NOT be decomposed
- Focus on Florida sales and use tax (Chapter 212) unless the query clearly involves other taxes
- Each sub-query should be self-contained and searchable
- Prioritize sub-queries by importance (1=highest, 5=lowest)

Return your response as valid JSON in this exact format:
{{
    "sub_queries": [
        {{"text": "specific search query text", "type": "definition|statute|rule|exemption|local|temporal", "priority": 1}}
    ],
    "reasoning": "Brief explanation of why you decomposed this way (or why you didn't)",
    "is_simple": false
}}

If the query is simple and doesn't need decomposition, return:
{{
    "sub_queries": [],
    "reasoning": "This is a simple, direct question that doesn't require decomposition",
    "is_simple": true
}}

User Query: {query}'''

CLASSIFICATION_PROMPT = '''Classify this Florida tax question into ONE primary category:

Categories:
- definition: Asks what something means under tax law (e.g., "What is tangible personal property?")
- exemption: Asks about tax exemptions (e.g., "Is food exempt from sales tax?")
- rate: Asks about tax rates (e.g., "What is the sales tax rate?")
- procedure: Asks about filing or compliance procedures (e.g., "How do I register for sales tax?")
- penalty: Asks about penalties or interest (e.g., "What is the penalty for late filing?")
- local: Asks about county-specific taxes (e.g., "What is the Miami-Dade surtax?")
- temporal: Asks about timing or effective dates (e.g., "When did the rate change?")

Query: {query}

Return ONLY the category name (one word, lowercase).'''

# System message for consistent behavior
SYSTEM_MESSAGE = '''You are a Florida tax law expert assistant specializing in sales and use tax (Chapter 212 of Florida Statutes).

Your role is to help decompose complex tax questions into searchable sub-queries for a legal document retrieval system.

Key Florida tax resources you're familiar with:
- Florida Statutes Chapter 212 (Sales and Use Tax)
- Florida Administrative Code 12A-1 (DOR Rules)
- Technical Assistance Advisements (TAAs)
- Florida case law on tax matters

Be precise and focused on Florida-specific tax law.'''

# Relevance scoring prompt for agent nodes
RELEVANCE_PROMPT = '''Given this tax law question and a document excerpt,
rate the relevance from 0 to 10:

Question: {query}
Document Type: {doc_type}
Citation: {citation}
Document Text:
{text}

Consider:
- Does this directly answer the question?
- Is this the authoritative source (statute > rule > case > taa)?
- Is this current law or potentially superseded?

Return JSON only: {{"score": 0-10, "reasoning": "brief explanation"}}'''
