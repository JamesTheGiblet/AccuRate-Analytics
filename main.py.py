import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import json
from datetime import datetime, timedelta
import os
import spacy
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataCertaintyScorer:
    def __init__(self):
        self.credibility_database = self.load_credibility_database()
        self.fact_checking_patterns = self.load_fact_checking_patterns()
        
    def load_credibility_database(self):
        """Load known credible and non-credible sources"""
        return {
            'high_credibility': {
                'sources': ['reuters.com', 'ap.org', 'bbc.com', 'npr.org', 'wsj.com'],
                'score': 0.9
            },
            'medium_credibility': {
                'sources': ['cnn.com', 'foxnews.com', 'theguardian.com', 'nytimes.com'],
                'score': 0.7
            },
            'low_credibility': {
                'sources': ['blogspot.com', 'wordpress.com', 'medium.com', 'reddit.com'],
                'score': 0.4
            },
            'user_generated': {
                'sources': ['twitter.com', 'facebook.com', 'instagram.com'],
                'score': 0.3
            }
        }
    
    def load_fact_checking_patterns(self):
        """Patterns that indicate factual vs opinion content"""
        return {
            'factual_indicators': [
                r'\d{4}',  # Years
                r'\$\d+',  # Monetary amounts
                r'\d+%',   # Percentages
                r'according to (study|research|data)',
                r'studies show',
                r'research indicates',
                r'data suggests'
            ],
            'opinion_indicators': [
                r'i think',
                r'in my opinion',
                r'i believe',
                r'it seems',
                r'probably',
                r'might be',
                r'could be'
            ],
            'exaggeration_indicators': [
                r'amazing',
                r'incredible',
                r'unbelievable',
                r'revolutionary',
                r'game-changing',
                r'mind-blowing'
            ]
        }
    
    def calculate_source_credibility(self, url, source):
        """Calculate credibility score based on source reputation"""
        max_score = 0.5  # Source credibility max weight
        
        # Check against known sources
        for credibility_level, data in self.credibility_database.items():
            for credible_source in data['sources']:
                if credible_source in url.lower():
                    return data['score'] * max_score
        
        # Domain-based scoring
        domain_extension_score = {
            '.gov': 0.9, '.edu': 0.8, '.org': 0.7, 
            '.com': 0.6, '.net': 0.5
        }
        
        for ext, score in domain_extension_score.items():
            if ext in url.lower():
                return score * max_score
        
        return 0.5 * max_score  # Default medium score
    
    def calculate_content_quality(self, title, content):
        """Calculate content quality score"""
        max_score = 0.3  # Content quality max weight
        
        text = f"{title} {content}".lower()
        score = 0.0
        
        # Length quality
        content_length = len(content.split())
        if content_length > 200:
            score += 0.1
        elif content_length > 50:
            score += 0.05
        
        # Factual indicators
        factual_count = sum(1 for pattern in self.fact_checking_patterns['factual_indicators'] 
                          if re.search(pattern, text))
        score += min(factual_count * 0.02, 0.1)
        
        # Opinion indicators (negative)
        opinion_count = sum(1 for pattern in self.fact_checking_patterns['opinion_indicators'] 
                           if re.search(pattern, text))
        score -= min(opinion_count * 0.01, 0.05)
        
        # Exaggeration indicators (negative)
        exaggeration_count = sum(1 for pattern in self.fact_checking_patterns['exaggeration_indicators'] 
                                if re.search(pattern, text))
        score -= min(exaggeration_count * 0.015, 0.05)
        
        return max(score, 0) * max_score
    
    def calculate_recency_score(self, date_string):
        """Calculate score based on content recency"""
        max_score = 0.2  # Recency max weight
        
        try:
            if isinstance(date_string, str):
                # Parse various date formats
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%B %d, %Y']:
                    try:
                        content_date = datetime.strptime(date_string, fmt)
                        break
                    except:
                        continue
                else:
                    # If no format matches, assume recent
                    return 0.1 * max_score
            else:
                content_date = date_string
            
            days_old = (datetime.now() - content_date).days
            
            if days_old <= 1:
                return 1.0 * max_score
            elif days_old <= 7:
                return 0.8 * max_score
            elif days_old <= 30:
                return 0.6 * max_score
            elif days_old <= 365:
                return 0.4 * max_score
            else:
                return 0.2 * max_score
                
        except:
            return 0.1 * max_score
    
    def calculate_sentiment_consistency(self, sentiment_scores):
        """Calculate consistency of sentiment analysis"""
        max_score = 0.1  # Sentiment consistency max weight
        
        if len(sentiment_scores) < 2:
            return 0.05 * max_score
        
        # Check if sentiments are consistent
        compound_scores = [s['compound'] for s in sentiment_scores]
        sentiment_std = np.std(compound_scores)
        
        if sentiment_std < 0.2:
            return 0.9 * max_score
        elif sentiment_std < 0.4:
            return 0.6 * max_score
        else:
            return 0.3 * max_score
    
    def calculate_cross_verification_score(self, data_points):
        """Calculate score based on cross-verification across sources"""
        max_score = 0.3  # Cross-verification max weight
        
        if len(data_points) < 2:
            return 0.1 * max_score
        
        # Check for similar facts across sources
        key_phrases = []
        for point in data_points:
            text = f"{point.get('title', '')} {point.get('content', '')}"
            phrases = self.extract_key_phrases(text)
            key_phrases.extend(phrases)
        
        # Count phrase frequency
        phrase_counts = Counter(key_phrases)
        verified_phrases = sum(1 for count in phrase_counts.values() if count > 1)
        
        verification_ratio = verified_phrases / len(phrase_counts) if phrase_counts else 0
        
        return min(verification_ratio * max_score, max_score)
    
    def extract_key_phrases(self, text):
        """Extract key factual phrases from text"""
        # Simple implementation - can be enhanced with NLP
        sentences = re.split(r'[.!?]+', text)
        key_phrases = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            if 5 <= len(words) <= 15:  # Reasonable phrase length
                # Look for factual statements
                if any(word in sentence.lower() for word in ['is', 'are', 'was', 'were', 'has', 'have']):
                    key_phrases.append(sentence.strip()[:100])  # Limit length
        
        return key_phrases[:5]  # Return top 5 phrases
    
    def calculate_total_certainty_score(self, data_point, all_data=[]):
        """Calculate overall certainty score for a data point"""
        url = data_point.get('url', '')
        source = data_point.get('source', '')
        title = data_point.get('title', '')
        content = data_point.get('content', '')
        date = data_point.get('date', '')
        sentiment_scores = [data_point.get('sentiment', {})]
        
        scores = {
            'source_credibility': self.calculate_source_credibility(url, source),
            'content_quality': self.calculate_content_quality(title, content),
            'recency': self.calculate_recency_score(date),
            'sentiment_consistency': self.calculate_sentiment_consistency(sentiment_scores),
            'cross_verification': self.calculate_cross_verification_score(all_data)
        }
        
        total_score = sum(scores.values())
        
        # Add bonus for high individual scores
        if scores['source_credibility'] > 0.3:
            total_score += 0.1
        if scores['content_quality'] > 0.15:
            total_score += 0.05
        
        return min(total_score, 1.0), scores

class EnhancedDataValidator:
    def __init__(self):
        self.certainty_scorer = DataCertaintyScorer()
        self.fact_checking_api = FactCheckingAPI()
        
    def validate_factual_claims(self, text):
        """Validate specific factual claims in text"""
        claims = self.extract_claims(text)
        validation_results = []
        
        for claim in claims:
            result = {
                'claim': claim,
                'is_verifiable': self.is_claim_verifiable(claim),
                'certainty_score': self.estimate_claim_certainty(claim),
                'supporting_evidence': [],
                'contradicting_evidence': []
            }
            validation_results.append(result)
        
        return validation_results
    
    def extract_claims(self, text):
        """Extract factual claims from text"""
        # Simple claim extraction - can be enhanced with NLP
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        claim_indicators = [
            r'\d+',  # Numbers
            r'is the', r'are the', r'was the', r'were the',
            r'has been', r'have been',
            r'according to', r'studies show', r'research indicates'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(re.search(indicator, sentence.lower()) for indicator in claim_indicators):
                if len(sentence.split()) >= 3:  # Minimum claim length
                    claims.append(sentence)
        
        return claims[:10]  # Limit to top 10 claims
    
    def is_claim_verifiable(self, claim):
        """Check if a claim is potentially verifiable"""
        non_verifiable_indicators = [
            'i think', 'i believe', 'in my opinion', 'probably',
            'might be', 'could be', 'seems to be'
        ]
        
        return not any(indicator in claim.lower() for indicator in non_verifiable_indicators)
    
    def estimate_claim_certainty(self, claim):
        """Estimate certainty level of a specific claim"""
        certainty_indicators = {
            'high_certainty': [r'\d{4}', r'\$\d+', r'\d+%', 'study', 'research', 'data'],
            'medium_certainty': ['report', 'analysis', 'findings'],
            'low_certainty': ['some', 'many', 'few', 'several']
        }
        
        score = 0.5  # Base score
        
        for level, indicators in certainty_indicators.items():
            for indicator in indicators:
                if re.search(indicator, claim.lower()):
                    if level == 'high_certainty':
                        score += 0.2
                    elif level == 'medium_certainty':
                        score += 0.1
                    elif level == 'low_certainty':
                        score -= 0.1
        
        return max(0.1, min(score, 1.0))

class FactCheckingAPI:
    """Simulated fact-checking API integration"""
    def __init__(self):
        self.known_facts = self.load_known_facts()
    
    def load_known_facts(self):
        """Load database of known facts for verification"""
        # In real implementation, this would connect to fact-checking APIs
        return {
            "climate change is real": {"certainty": 0.95, "sources": ["NASA", "IPCC"]},
            "vaccines are effective": {"certainty": 0.92, "sources": ["WHO", "CDC"]},
            "the earth is round": {"certainty": 0.99, "sources": ["Scientific consensus"]}
        }
    
    def check_claim(self, claim):
        """Check claim against known facts database"""
        claim_lower = claim.lower()
        
        for known_fact, data in self.known_facts.items():
            if known_fact in claim_lower:
                return {
                    'match': True,
                    'certainty': data['certainty'],
                    'sources': data['sources'],
                    'status': 'verified'
                }
        
        return {
            'match': False,
            'certainty': 0.5,
            'sources': [],
            'status': 'unknown'
        }

class EnhancedIntelligentWebScraper:
    def __init__(self):
        self.certainty_scorer = DataCertaintyScorer()
        self.data_validator = EnhancedDataValidator()
        self.scraped_data = []
        
    def enhance_data_with_certainty(self, data):
        """Add certainty scoring to all data points"""
        enhanced_data = []
        
        for data_point in data:
            certainty_score, score_breakdown = self.certainty_scorer.calculate_total_certainty_score(
                data_point, data
            )
            
            # Add certainty information to data point
            enhanced_point = data_point.copy()
            enhanced_point['certainty_score'] = certainty_score
            enhanced_point['certainty_breakdown'] = score_breakdown
            enhanced_point['certainty_level'] = self.get_certainty_level(certainty_score)
            enhanced_point['data_quality'] = self.assess_data_quality(score_breakdown)
            
            # Add fact validation for high-impact claims
            if certainty_score > 0.6:
                text = f"{data_point.get('title', '')} {data_point.get('content', '')}"
                enhanced_point['fact_validation'] = self.data_validator.validate_factual_claims(text)
            
            enhanced_data.append(enhanced_point)
        
        return enhanced_data
    
    def get_certainty_level(self, score):
        """Convert numerical score to certainty level"""
        if score >= 0.8:
            return "Very High"
        elif score >= 0.7:
            return "High"
        elif score >= 0.6:
            return "Medium-High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.4:
            return "Medium-Low"
        elif score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def assess_data_quality(self, score_breakdown):
        """Assess overall data quality based on score breakdown"""
        quality_indicators = []
        
        if score_breakdown['source_credibility'] > 0.3:
            quality_indicators.append("Credible Source")
        if score_breakdown['content_quality'] > 0.15:
            quality_indicators.append("High-Quality Content")
        if score_breakdown['recency'] > 0.1:
            quality_indicators.append("Recent Information")
        if score_breakdown['cross_verification'] > 0.15:
            quality_indicators.append("Cross-Verified")
        
        return quality_indicators if quality_indicators else ["Standard Quality"]
    
    def generate_certainty_report(self, enhanced_data):
        """Generate comprehensive certainty analysis report"""
        certainty_scores = [point['certainty_score'] for point in enhanced_data]
        
        report = {
            'overview': {
                'total_data_points': len(enhanced_data),
                'average_certainty': np.mean(certainty_scores) if certainty_scores else 0,
                'median_certainty': np.median(certainty_scores) if certainty_scores else 0,
                'certainty_std': np.std(certainty_scores) if certainty_scores else 0
            },
            'distribution': {
                'very_high': sum(1 for s in certainty_scores if s >= 0.8),
                'high': sum(1 for s in certainty_scores if 0.7 <= s < 0.8),
                'medium_high': sum(1 for s in certainty_scores if 0.6 <= s < 0.7),
                'medium': sum(1 for s in certainty_scores if 0.5 <= s < 0.6),
                'medium_low': sum(1 for s in certainty_scores if 0.4 <= s < 0.5),
                'low': sum(1 for s in certainty_scores if 0.3 <= s < 0.4),
                'very_low': sum(1 for s in certainty_scores if s < 0.3)
            },
            'top_high_certainty_sources': self.get_top_sources_by_certainty(enhanced_data, min_score=0.7),
            'reliability_warnings': self.generate_reliability_warnings(enhanced_data),
            'quality_metrics': self.calculate_quality_metrics(enhanced_data)
        }
        
        return report
    
    def get_top_sources_by_certainty(self, data, min_score=0.7):
        """Get top sources with high certainty scores"""
        high_certainty_sources = {}
        
        for point in data:
            if point['certainty_score'] >= min_score:
                source = point.get('source', 'Unknown')
                if source not in high_certainty_sources:
                    high_certainty_sources[source] = []
                high_certainty_sources[source].append(point['certainty_score'])
        
        # Calculate average certainty per source
        source_avg_certainty = {}
        for source, scores in high_certainty_sources.items():
            source_avg_certainty[source] = np.mean(scores)
        
        return dict(sorted(source_avg_certainty.items(), 
                          key=lambda x: x[1], reverse=True)[:5])
    
    def generate_reliability_warnings(self, data):
        """Generate warnings about low-reliability data"""
        warnings = []
        low_certainty_items = [point for point in data if point['certainty_score'] < 0.4]
        
        if len(low_certainty_items) > len(data) * 0.3:
            warnings.append(f"⚠️ {len(low_certainty_items)} low-certainty items detected ({len(low_certainty_items)/len(data)*100:.1f}% of total)")
        
        # Check for unverified major claims
        for point in data:
            if point['certainty_score'] > 0.7 and 'fact_validation' in point:
                unverified_claims = [claim for claim in point['fact_validation'] 
                                   if not claim['is_verifiable']]
                if unverified_claims:
                    warnings.append(f"⚠️ Unverifiable claims in high-certainty source: {point.get('source', 'Unknown')}")
        
        return warnings if warnings else ["✅ No major reliability issues detected"]
    
    def calculate_quality_metrics(self, data):
        """Calculate various data quality metrics"""
        metrics = {
            'source_diversity': len(set(point.get('source', '') for point in data)),
            'recency_distribution': self.analyze_recency_distribution(data),
            'content_completeness': sum(1 for point in data if len(point.get('content', '').split()) > 50) / len(data),
            'cross_verification_rate': np.mean([point['certainty_breakdown']['cross_verification'] for point in data])
        }
        
        return metrics
    
    def analyze_recency_distribution(self, data):
        """Analyze the distribution of content recency"""
        recency_categories = {'recent': 0, 'recent_week': 0, 'recent_month': 0, 'older': 0}
        
        for point in data:
            recency_score = point['certainty_breakdown']['recency']
            if recency_score > 0.15:
                recency_categories['recent'] += 1
            elif recency_score > 0.12:
                recency_categories['recent_week'] += 1
            elif recency_score > 0.08:
                recency_categories['recent_month'] += 1
            else:
                recency_categories['older'] += 1
        
        return recency_categories

# Enhanced main application with certainty scoring
class CertaintyAwareWebAnalyzer(EnhancedIntelligentWebScraper):
    def __init__(self):
        super().__init__()
        
    def generate_enhanced_report(self, user_input, intent_data, raw_data):
        """Generate report with certainty awareness"""
        # Enhance data with certainty scoring
        enhanced_data = self.enhance_data_with_certainty(raw_data)
        
        # Generate certainty report
        certainty_report = self.generate_certainty_report(enhanced_data)
        
        # Filter high-certainty data for primary analysis
        high_certainty_data = [point for point in enhanced_data if point['certainty_score'] >= 0.6]
        medium_certainty_data = [point for point in enhanced_data if 0.4 <= point['certainty_score'] < 0.6]
        low_certainty_data = [point for point in enhanced_data if point['certainty_score'] < 0.4]
        
        report = f"""
# 🎯 INTELLIGENT WEB ANALYSIS WITH CERTAINTY SCORING

## Data Reliability Assessment
**Overall Certainty Score**: {certainty_report['overview']['average_certainty']:.2f}/1.0

### Certainty Distribution:
- 🔴 Very High Certainty: {certainty_report['distribution']['very_high']} items
- 🟢 High Certainty: {certainty_report['distribution']['high']} items
- 🟡 Medium-High Certainty: {certainty_report['distribution']['medium_high']} items
- 🟠 Medium Certainty: {certainty_report['distribution']['medium']} items
- 🟣 Medium-Low Certainty: {certainty_report['distribution']['medium_low']} items
- 🔵 Low Certainty: {certainty_report['distribution']['low']} items
- ⚫ Very Low Certainty: {certainty_report['distribution']['very_low']} items

### Reliability Warnings:
{"".join([f"- {warning}\\n" for warning in certainty_report['reliability_warnings']])}

## High-Certainty Insights (Score ≥ 0.6)
*Based on {len(high_certainty_data)} verified data points*

{self.generate_high_certainty_insights(high_certainty_data, intent_data)}

## Medium-Certainty Context (Score 0.4-0.6)
*Supplementary information from {len(medium_certainty_data)} sources*

{self.generate_medium_certainty_context(medium_certainty_data)}

## ⚠️ Low-Certainty Notes (Score < 0.4)
*Use with caution - {len(low_certainty_data)} unverified items*

{self.generate_low_certainty_notes(low_certainty_data)}

## Data Quality Metrics
- **Source Diversity**: {certainty_report['quality_metrics']['source_diversity']} unique sources
- **Content Completeness**: {certainty_report['quality_metrics']['content_completeness']:.1%}
- **Cross-Verification Rate**: {certainty_report['quality_metrics']['cross_verification_rate']:.2f}/1.0

## Top Reliable Sources
{self.format_reliable_sources(certainty_report['top_high_certainty_sources'])}

---
*Report generated with certainty-aware analysis on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return report, enhanced_data, certainty_report
    
    def generate_high_certainty_insights(self, high_certainty_data, intent_data):
        """Generate insights from high-certainty data"""
        if not high_certainty_data:
            return "No high-certainty data available for reliable insights."
        
        # Analyze high-certainty data specifically
        insights = "### Key Verified Findings:\\n"
        
        # Sample the most certain points
        top_certain_points = sorted(high_certainty_data, 
                                  key=lambda x: x['certainty_score'], reverse=True)[:3]
        
        for i, point in enumerate(top_certain_points, 1):
            insights += f"{i}. **{point.get('title', 'No title')}**  \n"
            insights += f"   - Certainty: {point['certainty_score']:.2f} ({point['certainty_level']})  \n"
            insights += f"   - Source: {point.get('source', 'Unknown')}  \n"
            insights += f"   - Quality: {', '.join(point['data_quality'])}  \n\n"
        
        return insights
    
    def generate_medium_certainty_context(self, medium_certainty_data):
        """Generate context from medium-certainty data"""
        if not medium_certainty_data:
            return "No medium-certainty context available."
        
        return f"Found {len(medium_certainty_data)} items with moderate reliability that provide additional context but should be verified independently."
    
    def generate_low_certainty_notes(self, low_certainty_data):
        """Generate notes about low-certainty data"""
        if not low_certainty_data:
            return "No low-certainty data detected."
        
        low_certainty_sources = Counter([point.get('source', 'Unknown') for point in low_certainty_data])
        
        notes = "The following sources showed lower reliability scores:\\n"
        for source, count in low_certainty_sources.most_common(3):
            notes += f"- {source}: {count} items\\n"
        
        notes += "\\n*Consider verifying these findings with more reliable sources.*"
        return notes
    
    def format_reliable_sources(self, reliable_sources):
        """Format reliable sources information"""
        if not reliable_sources:
            return "No highly reliable sources identified in this dataset."
        
        formatted = ""
        for source, avg_certainty in reliable_sources.items():
            formatted += f"- **{source}**: {avg_certainty:.2f} average certainty score\\n"
        
        return formatted

# Example usage
def main():
    analyzer = CertaintyAwareWebAnalyzer()
    
    # Simulate data collection
    user_input = "climate change impacts"
    intent_data = {'primary_intent': 'research_analysis', 'confidence': 0.85}
    
    # Simulated raw data (in real implementation, this comes from scraping)
    raw_data = [
        {
            'source': 'NASA Climate',
            'title': 'Scientific Consensus on Climate Change',
            'content': '97% of climate scientists agree that climate change is real and primarily caused by human activities. Global temperatures have risen 1.1°C since pre-industrial times.',
            'url': 'https://climate.nasa.gov',
            'date': '2024-01-15'
        },
        {
            'source': 'Personal Blog',
            'title': 'My thoughts on climate',
            'content': 'I think climate change might be exaggerated by media. Some scientists disagree with the mainstream view.',
            'url': 'https://personalblog.example.com',
            'date': '2024-01-10'
        }
    ]
    
    # Generate enhanced report
    report, enhanced_data, certainty_report = analyzer.generate_enhanced_report(
        user_input, intent_data, raw_data
    )
    
    print(report)
    
    # Display certainty scores for each item
    print("\n" + "="*60)
    print("📊 DETAILED CERTAINTY SCORES PER ITEM")
    print("="*60)
    
    for i, point in enumerate(enhanced_data, 1):
        print(f"\n{i}. {point.get('title', 'No title')}")
        print(f"   Source: {point.get('source', 'Unknown')}")
        print(f"   Certainty Score: {point['certainty_score']:.2f} ({point['certainty_level']})")
        print(f"   Quality Indicators: {', '.join(point['data_quality'])}")

if __name__ == "__main__":
    main()