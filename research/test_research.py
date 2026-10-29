import unittest,numpy as np
from analysis import motion_metrics,align_pairs,agreement,capture_analysis
from benchmark import cloud_features
from glove_capture import validate_values
class ResearchTests(unittest.TestCase):
 def test_marker_features_permutation_invariant(self):
  p=np.random.default_rng(1).normal(size=(8,3));np.testing.assert_allclose(cloud_features(p),cloud_features(p[::-1]),atol=1e-10)
 def test_zero_motion_has_zero_jerk(self):
  m=motion_metrics(np.arange(100)*70,np.zeros(100));self.assertAlmostEqual(m['rms_jerk_per_s3'],0);self.assertEqual(m['amplitude_p95_p05'],0)
 def test_gaps_rejected(self):
  t=np.arange(20)*70;t[10:]+=1000
  with self.assertRaises(ValueError):motion_metrics(t,np.zeros(20))
 def test_match_tolerance_and_no_reuse(self):
  c=[{'time_ms':v} for v in [0,1,1000]];g=[{'time_ms':0}];self.assertEqual(len(align_pairs(c,g,50)),1)
 def test_known_bias(self):
  pairs=[({'values':{'index':v+.1}},{'values':{'index':v}},0) for v in [.1,.2,.3]];r=agreement(pairs)['fingers']['index'];self.assertAlmostEqual(r['mae'],.1);self.assertAlmostEqual(r['correlation'],1)
 def test_failures_kept_in_denominator(self):
  d={'schema':'qolda-vision.capture.v1','source':'camera','meta':{'task':'other'},'rows':[{'detected':False,'inference_ms':5},{'detected':True,'inference_ms':5}]};self.assertEqual(capture_analysis(d)['missing_hand_rate'],.5)
 def test_glove_raw_adc_rejected(self):
  with self.assertRaises(ValueError):validate_values({'values':{'index':1023}})
if __name__=='__main__':unittest.main()
