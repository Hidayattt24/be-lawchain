"""
Performance Comparison Script: Original vs Optimized LangChain
"""

import time
import json
import statistics
from typing import Dict, List, Any
from datetime import datetime


class LawChainPerformanceComparator:
    """Compare performance between original and optimized LangChain implementations"""
    
    def __init__(self):
        self.test_questions = [
            "Apa itu Pancasila menurut UUD 1945?",
            "Bagaimana tugas dan wewenang Presiden Republik Indonesia?",
            "Sebutkan hak asasi manusia yang dijamin dalam UUD 1945",
            "Jelaskan tentang Majelis Permusyawaratan Rakyat (MPR)",
            "Apa saja kewajiban warga negara menurut UUD 1945?",
            "Bagaimana sistem pemerintahan Indonesia menurut UUD 1945?",
            "Jelaskan tentang Mahkamah Konstitusi dan tugasnya",
            "Apa yang dimaksud dengan Negara Kesatuan Republik Indonesia?"
        ]
        
        self.results = {
            'original': [],
            'optimized': []
        }
    
    def test_original_langchain(self):
        """Test original LangChain implementation"""
        print("🔄 Testing Original LangChain Implementation...")
        print("=" * 60)
        
        try:
            from app.services.lawchain_indonesia import LawChainIndonesia
            
            print("📂 Initializing original LangChain...")
            lawchain = LawChainIndonesia()
            lawchain.initialize()
            
            for i, question in enumerate(self.test_questions, 1):
                print(f"\n--- Original Test {i}: {question[:50]}... ---")
                
                try:
                    start_time = time.time()
                    response = lawchain.ask_question_with_custom_qa(question)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    
                    result = {
                        'question': question,
                        'processing_time': processing_time,
                        'accuracy': response['metrics']['estimated_accuracy'],
                        'confidence': response['metrics']['confidence_score'],
                        'sources_count': response['jumlah_sumber'],
                        'answer_length': len(response['jawaban']),
                        'semantic_similarity': response['metrics']['semantic_similarity'],
                        'answer_relevance': response['metrics']['answer_relevance'],
                        'source_quality': response['metrics']['source_quality'],
                        'success': True
                    }
                    
                    self.results['original'].append(result)
                    
                    print(f"✅ Success - Time: {processing_time:.2f}s, Accuracy: {result['accuracy']:.1f}%")
                    
                except Exception as e:
                    print(f"❌ Failed: {str(e)}")
                    self.results['original'].append({
                        'question': question,
                        'error': str(e),
                        'success': False
                    })
                    
        except Exception as e:
            print(f"❌ Original LangChain initialization failed: {str(e)}")
            return False
        
        return True
    
    def test_optimized_langchain(self):
        """Test optimized LangChain implementation"""
        print("\n🚀 Testing Optimized LangChain Implementation...")
        print("=" * 60)
        
        try:
            from app.services.lawchain_optimized import OptimizedLawChainIndonesia
            
            print("📂 Initializing optimized LangChain...")
            lawchain = OptimizedLawChainIndonesia()
            lawchain.initialize_optimized()
            
            for i, question in enumerate(self.test_questions, 1):
                print(f"\n--- Optimized Test {i}: {question[:50]}... ---")
                
                try:
                    start_time = time.time()
                    response = lawchain.ask_question_optimized(question)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    
                    result = {
                        'question': question,
                        'processing_time': processing_time,
                        'accuracy': response['metrics']['estimated_accuracy'],
                        'confidence': response['metrics']['confidence_score'],
                        'sources_count': response['jumlah_sumber'],
                        'answer_length': len(response['jawaban']),
                        'semantic_similarity': response['metrics']['semantic_similarity'],
                        'answer_relevance': response['metrics']['answer_relevance'],
                        'source_quality': response['metrics']['source_quality'],
                        'success': True
                    }
                    
                    self.results['optimized'].append(result)
                    
                    print(f"✅ Success - Time: {processing_time:.2f}s, Accuracy: {result['accuracy']:.1f}%")
                    
                except Exception as e:
                    print(f"❌ Failed: {str(e)}")
                    self.results['optimized'].append({
                        'question': question,
                        'error': str(e),
                        'success': False
                    })
                    
        except Exception as e:
            print(f"❌ Optimized LangChain initialization failed: {str(e)}")
            return False
        
        return True
    
    def calculate_statistics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate performance statistics"""
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'success_rate': 0.0,
                'avg_processing_time': 0.0,
                'avg_accuracy': 0.0,
                'avg_confidence': 0.0,
                'avg_sources_count': 0.0,
                'avg_answer_length': 0.0,
                'avg_semantic_similarity': 0.0,
                'avg_answer_relevance': 0.0,
                'avg_source_quality': 0.0
            }
        
        return {
            'success_rate': len(successful_results) / len(results) * 100,
            'avg_processing_time': statistics.mean([r['processing_time'] for r in successful_results]),
            'avg_accuracy': statistics.mean([r['accuracy'] for r in successful_results]),
            'avg_confidence': statistics.mean([r['confidence'] for r in successful_results]),
            'avg_sources_count': statistics.mean([r['sources_count'] for r in successful_results]),
            'avg_answer_length': statistics.mean([r['answer_length'] for r in successful_results]),
            'avg_semantic_similarity': statistics.mean([r['semantic_similarity'] for r in successful_results]),
            'avg_answer_relevance': statistics.mean([r['answer_relevance'] for r in successful_results]),
            'avg_source_quality': statistics.mean([r['source_quality'] for r in successful_results])
        }
    
    def display_comparison_report(self):
        """Display detailed comparison report"""
        print("\n" + "=" * 80)
        print("📊 LAWCHAIN PERFORMANCE COMPARISON REPORT")
        print("=" * 80)
        
        original_stats = self.calculate_statistics(self.results['original'])
        optimized_stats = self.calculate_statistics(self.results['optimized'])
        
        print(f"\n🔍 TEST SUMMARY:")
        print(f"   Total Test Questions: {len(self.test_questions)}")
        print(f"   Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Success Rate Comparison
        print(f"\n📈 SUCCESS RATE COMPARISON:")
        print(f"   Original LangChain:  {original_stats['success_rate']:.1f}%")
        print(f"   Optimized LangChain: {optimized_stats['success_rate']:.1f}%")
        if optimized_stats['success_rate'] > original_stats['success_rate']:
            improvement = optimized_stats['success_rate'] - original_stats['success_rate']
            print(f"   🚀 Improvement: +{improvement:.1f}%")
        
        # Performance Metrics Comparison
        print(f"\n⚡ PERFORMANCE METRICS COMPARISON:")
        
        metrics = [
            ('Processing Time (avg)', 'avg_processing_time', 's', True),  # Lower is better
            ('Accuracy (avg)', 'avg_accuracy', '%', False),  # Higher is better
            ('Confidence (avg)', 'avg_confidence', '%', False),  # Higher is better
            ('Sources Count (avg)', 'avg_sources_count', '', False),  # Higher is better
            ('Answer Length (avg)', 'avg_answer_length', 'chars', False),  # Higher is better
            ('Semantic Similarity (avg)', 'avg_semantic_similarity', '%', False),  # Higher is better
            ('Answer Relevance (avg)', 'avg_answer_relevance', '%', False),  # Higher is better
            ('Source Quality (avg)', 'avg_source_quality', '', False)  # Higher is better
        ]
        
        for metric_name, metric_key, unit, lower_is_better in metrics:
            original_value = original_stats[metric_key]
            optimized_value = optimized_stats[metric_key]
            
            if metric_key == 'avg_processing_time':
                print(f"\n   {metric_name}:")
                print(f"      Original:  {original_value:.2f}{unit}")
                print(f"      Optimized: {optimized_value:.2f}{unit}")
            else:
                print(f"\n   {metric_name}:")
                print(f"      Original:  {original_value:.1f}{unit}")
                print(f"      Optimized: {optimized_value:.1f}{unit}")
            
            # Calculate improvement
            if original_value > 0:
                if lower_is_better:
                    improvement = ((original_value - optimized_value) / original_value) * 100
                    if improvement > 0:
                        print(f"      🚀 Improvement: -{improvement:.1f}% (faster)")
                    elif improvement < 0:
                        print(f"      📉 Regression: +{abs(improvement):.1f}% (slower)")
                else:
                    improvement = ((optimized_value - original_value) / original_value) * 100
                    if improvement > 0:
                        print(f"      🚀 Improvement: +{improvement:.1f}%")
                    elif improvement < 0:
                        print(f"      📉 Regression: {improvement:.1f}%")
        
        # Overall Assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        improvements = 0
        regressions = 0
        
        for metric_name, metric_key, unit, lower_is_better in metrics:
            original_value = original_stats[metric_key]
            optimized_value = optimized_stats[metric_key]
            
            if original_value > 0:
                if lower_is_better:
                    if optimized_value < original_value:
                        improvements += 1
                    elif optimized_value > original_value:
                        regressions += 1
                else:
                    if optimized_value > original_value:
                        improvements += 1
                    elif optimized_value < original_value:
                        regressions += 1
        
        print(f"   Metrics Improved: {improvements}/{len(metrics)}")
        print(f"   Metrics Regressed: {regressions}/{len(metrics)}")
        
        if improvements > regressions:
            print("   🎉 OPTIMIZATION SUCCESSFUL! Performance improved overall.")
        elif improvements == regressions:
            print("   ⚖️ MIXED RESULTS: Some improvements, some regressions.")
        else:
            print("   ⚠️ OPTIMIZATION NEEDS REVIEW: More regressions than improvements.")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if optimized_stats['avg_processing_time'] < original_stats['avg_processing_time']:
            print("   ✅ Processing time improved - optimizations working well")
        else:
            print("   ⚠️ Processing time increased - consider further chunk size optimization")
            
        if optimized_stats['avg_accuracy'] > original_stats['avg_accuracy']:
            print("   ✅ Accuracy improved - better retrieval and context filtering")
        else:
            print("   ⚠️ Accuracy decreased - review prompt template and MMR parameters")
            
        if optimized_stats['avg_sources_count'] <= 5:
            print("   ✅ Source count optimized - good focus on quality over quantity")
        else:
            print("   ⚠️ Too many sources retrieved - reduce max_retrieval_docs")
        
        print("\n" + "=" * 80)
    
    def save_detailed_results(self):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_comparison_{timestamp}.json"
        
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'test_questions': self.test_questions,
            'original_results': self.results['original'],
            'optimized_results': self.results['optimized'],
            'original_stats': self.calculate_statistics(self.results['original']),
            'optimized_stats': self.calculate_statistics(self.results['optimized'])
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            print(f"📄 Detailed results saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")
    
    def run_full_comparison(self):
        """Run complete performance comparison"""
        print("🏛️ LawChain Performance Comparison Tool")
        print("=" * 80)
        print("This tool will compare Original vs Optimized LangChain implementations")
        print("⚠️ Warning: This may take 10-20 minutes depending on system performance")
        print()
        
        # Test original implementation
        original_success = self.test_original_langchain()
        
        # Test optimized implementation  
        optimized_success = self.test_optimized_langchain()
        
        if original_success or optimized_success:
            # Display comparison report
            self.display_comparison_report()
            
            # Save detailed results
            self.save_detailed_results()
        else:
            print("❌ Both implementations failed. Cannot generate comparison report.")


def main():
    """Main function"""
    try:
        comparator = LawChainPerformanceComparator()
        comparator.run_full_comparison()
    except KeyboardInterrupt:
        print("\n⚠️ Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Comparison failed: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
