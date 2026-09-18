"""BS contracts: no target leakage, cloud handling, severity and RLE round trip."""
import unittest

import numpy as np

from firemon.bs_model import extract_features, decode
from firemon.bs_rules import Thresholds
from firemon.io import BSChip
from firemon.rle import encode, decode as rle_decode
from firemon.metric import Accumulator


def chip():
    rng=np.random.default_rng(3)
    pre=rng.integers(100,6000,(32,40,10),dtype=np.uint16)
    post=pre.copy()
    pre[...,9]=post[...,9]=4
    post[...,6]=500
    post[...,8]=4000
    sar=rng.integers(-2500,-500,(32,40,2),dtype=np.int16)
    aux=np.zeros((32,40,3),np.int16);aux[...,2]=30
    return BSChip('example',pre,post,sar,sar.copy(),aux,np.zeros((32,40),np.uint8))


class BSTests(unittest.TestCase):
    def test_features_do_not_read_labels_or_id(self):
        c=chip();a,names=extract_features(c)
        c.mask[:]=3;c.chip_id='unseen';b,other=extract_features(c)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(names,other)
        self.assertEqual(a.shape[:2],(32,40))
        self.assertEqual(a.shape[-1],len(set(names)))
        self.assertTrue(np.isfinite(a).all())

    def test_decode_cloud_and_nonoverlap(self):
        c=chip();c.s2_pre[0,:,9]=3;c.s2_post[1,:,9]=9
        th=Thresholds({'default':[.1,.2,.4]})
        pred=decode(np.ones((32,40),np.float32),c,th,.5)
        self.assertFalse(pred[:2].any())
        self.assertTrue(set(np.unique(pred)) <= {0,1,2,3})
        restored=np.zeros_like(pred)
        for k in (1,2,3):restored[rle_decode(encode(pred==k),pred.shape)]=k
        np.testing.assert_array_equal(restored,pred)

    def test_all_cloud_is_empty(self):
        c=chip();c.s2_post[...,9]=9
        p=decode(np.ones((32,40),np.float32),c,Thresholds({'default':[.1,.2,.4]}),min_size=10,smooth=3)
        self.assertFalse(p.any())

    def test_micro_metric_known_counts(self):
        a=Accumulator();a.add_bs(np.array([0,1,2,3]),np.array([0,1,3,3]))
        r=a.result()
        self.assertEqual(r['IoU_burn'],1)
        self.assertAlmostEqual(r['mIoU_sev'],.5)


if __name__=='__main__':unittest.main()
