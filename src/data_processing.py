import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SanskritCorpusProcessor:
    def __init__(self, data_dir: str = "/home/sai/Desktop/ullu/data/1_sanskr"):
        self.data_dir = Path(data_dir)
        self.passages = []
        
    def extract_from_html(self, file_path: Path) -> List[Dict]:
        """Extract verses/passages from HTML files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text and split into lines
            text = soup.get_text()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            passages = []
            current_passage = []
            
            for line in lines:
                # Skip very short lines or numbers
                if len(line) < 10 or line.isdigit():
                    continue
                    
                # Sanskrit verse typically has 2-4 lines
                current_passage.append(line)
                
                # If we have enough lines, create a passage
                if len(current_passage) >= 2:
                    passage_text = ' '.join(current_passage)
                    
                    # Basic Sanskrit text detection
                    if self._is_sanskrit_text(passage_text):
                        passages.append({
                            'text': passage_text,
                            'source_file': file_path.name,
                            'category': self._get_category_from_path(file_path),
                            'length': len(passage_text)
                        })
                    
                    current_passage = []
            
            return passages
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def extract_from_xml(self, file_path: Path) -> List[Dict]:
        """Extract verses/passages from TEI XML files."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            passages = []
            
            # Handle different XML structures
            for elem in root.iter():
                if elem.text and len(elem.text.strip()) > 20:
                    text = elem.text.strip()
                    
                    if self._is_sanskrit_text(text):
                        passages.append({
                            'text': text,
                            'source_file': file_path.name,
                            'category': 'tei',
                            'element': elem.tag,
                            'length': len(text)
                        })
            
            return passages
            
        except Exception as e:
            logger.error(f"Error processing XML {file_path}: {e}")
            return []
    
    def _is_sanskrit_text(self, text: str) -> bool:
        """Basic heuristic to detect Sanskrit text."""
        # Check for Devanagari characters
        devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
        
        # Check for IAST characters
        iast_chars = re.findall(r'[āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ]', text.lower())
        
        # Check for Sanskrit-like patterns
        sanskrit_patterns = [
            r'atha\b', r'eva\b', r'ca\b', r'tu\b', r'hi\b',
            r'iti\b', r'yat\b', r'tad\b', r'sarvam?\b'
        ]
        
        pattern_matches = sum(1 for pattern in sanskrit_patterns 
                            if re.search(pattern, text.lower()))
        
        # Text is likely Sanskrit if it has Devanagari, IAST chars, or Sanskrit patterns
        return (devanagari_count > len(text) * 0.1 or 
                len(iast_chars) > 3 or 
                pattern_matches > 1)
    
    def _get_category_from_path(self, file_path: Path) -> str:
        """Extract category from file path."""
        path_parts = file_path.parts
        for part in path_parts:
            if part.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_')):
                return part
        return 'unknown'
    
    def process_all_files(self) -> pd.DataFrame:
        """Process all files in the corpus directory."""
        logger.info("Starting corpus processing...")
        
        all_passages = []
        
        # Process HTML files
        for category_dir in self.data_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name == 'tei':
                for html_file in category_dir.glob('*.htm'):
                    passages = self.extract_from_html(html_file)
                    all_passages.extend(passages)
                    logger.info(f"Processed {html_file.name}: {len(passages)} passages")
        
        # Process TEI XML files
        tei_dir = self.data_dir / 'tei'
        if tei_dir.exists():
            for xml_file in tei_dir.glob('*.xml'):
                passages = self.extract_from_xml(xml_file)
                all_passages.extend(passages)
                logger.info(f"Processed {xml_file.name}: {len(passages)} passages")
        
        df = pd.DataFrame(all_passages)
        logger.info(f"Total passages extracted: {len(df)}")
        
        return df
    
    def create_quote_dataset(self, df: pd.DataFrame, num_quotes: int = 1000) -> pd.DataFrame:
        """Create a dataset of quotes for training/testing."""
        # Filter for appropriate length passages (likely complete verses)
        filtered_df = df[
            (df['length'] >= 50) & 
            (df['length'] <= 500)
        ].copy()
        
        # Sample quotes from different categories
        quotes = []
        
        categories = filtered_df['category'].unique()
        quotes_per_category = num_quotes // len(categories)
        
        for category in categories:
            category_df = filtered_df[filtered_df['category'] == category]
            
            if len(category_df) > quotes_per_category:
                sampled = category_df.sample(n=quotes_per_category, random_state=42)
            else:
                sampled = category_df
            
            quotes.extend(sampled.to_dict('records'))
        
        quote_df = pd.DataFrame(quotes)
        
        # Add metadata
        quote_df['quote_id'] = range(len(quote_df))
        quote_df['work'] = quote_df['source_file'].str.replace('.htm', '').str.replace('.xml', '')
        
        return quote_df

def main():
    processor = SanskritCorpusProcessor()
    
    # Process all files
    corpus_df = processor.process_all_files()
    
    # Save full corpus
    corpus_df.to_csv('/home/sai/Desktop/ullu/data/sanskrit_corpus.csv', index=False)
    
    # Create quote dataset
    quotes_df = processor.create_quote_dataset(corpus_df)
    quotes_df.to_csv('/home/sai/Desktop/ullu/data/sanskrit_quotes.csv', index=False)
    
    print(f"Processed {len(corpus_df)} total passages")
    print(f"Created {len(quotes_df)} quote samples")
    print(f"Categories: {corpus_df['category'].value_counts()}")

if __name__ == "__main__":
    main()