import unittest
import torch
import torch.nn as nn
import numpy as np

# Import modules under test
from src.data.preprocessing import log_compact, ZScoreNormalizer, mask_sequence
from src.models.mamba_encoder import MambaEncoder
from src.models.classifier_head import ClassifierHead
from src.models.parametric_umap import ParametricUMAPLoss
from src.memory.memory_bank import DeviceMemoryBank
from src.memory.threshold import DeviceThresholdManager
from src.gating.gate import AdaptiveInferenceGate
from src.quantization.quantize import log_quantize_dequantize, quantize_model, linear_quantize_dequantize
from src.runtime.pipeline import DBMambaRuntimePipeline, FastClassifier

class TestDBMambaComponents(unittest.TestCase):
    
    def setUp(self):
        # Setup common tensor sizes
        self.B = 4
        self.L = 16
        self.D_feat = 4
        self.d_model = 32
        
        # Random input sequence
        torch.manual_seed(42)
        self.dummy_input = torch.randn(self.B, self.L, self.D_feat) * 10.0 + 5.0
        # Clean negative values for lengths and IAT to represent network metrics
        self.dummy_input[:, :, 0] = torch.clamp(self.dummy_input[:, :, 0], min=40.0) # Packet lengths
        self.dummy_input[:, :, 2] = torch.clamp(self.dummy_input[:, :, 2], min=0.0)  # IAT

    def test_preprocessing(self):
        # Test log compaction
        compacted = log_compact(self.dummy_input)
        self.assertEqual(compacted.shape, self.dummy_input.shape)
        # Ensure values are compacted: log1p(x) <= x
        self.assertTrue(torch.all(compacted[:, :, 0] <= self.dummy_input[:, :, 0]))
        
        # Test ZScoreNormalizer
        normalizer = ZScoreNormalizer(num_features=self.D_feat)
        normalizer.fit(compacted)
        transformed = normalizer.transform(compacted)
        self.assertEqual(transformed.shape, compacted.shape)
        self.assertTrue(normalizer.is_fitted)
        
        # Test masking
        masked, orig, mask = mask_sequence(transformed, mask_ratio=0.15)
        self.assertEqual(masked.shape, transformed.shape)
        self.assertEqual(mask.shape, (self.B, self.L))
        # Masked positions should be zeroed out
        self.assertTrue(torch.all(masked[mask] == 0.0))
        # Unmasked positions should match original
        self.assertTrue(torch.all(masked[~mask] == transformed[~mask]))

    def test_mamba_encoder_shapes(self):
        encoder = MambaEncoder(num_features=self.D_feat, d_model=self.d_model, d_state=8, d_conv=2, num_layers=1)
        seq_embeddings, pooled_embedding = encoder(self.dummy_input)
        
        # Verify shapes
        self.assertEqual(seq_embeddings.shape, (self.B, self.L, self.d_model))
        self.assertEqual(pooled_embedding.shape, (self.B, self.d_model))

    def test_umap_loss_gradients_flow(self):
        encoder = MambaEncoder(num_features=self.D_feat, d_model=self.d_model, d_state=8, d_conv=2, num_layers=1)
        umap_loss_fn = ParametricUMAPLoss(k=2)
        
        # Run forward pass through encoder
        seq_emb, pooled_emb = encoder(self.dummy_input)
        
        # Compute UMAP loss
        loss = umap_loss_fn(self.dummy_input, pooled_emb)
        
        # Verify loss is a scalar and requires grad
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.requires_grad)
        
        # Verify gradient propagates to encoder parameters
        loss.backward()
        for name, param in encoder.named_parameters():
            if param.requires_grad and "input_projection" in name:
                self.assertIsNotNone(param.grad)
                # Check that gradients are not nan
                self.assertFalse(torch.isnan(param.grad).any())

    def test_memory_bank_clustering(self):
        # We generate embeddings
        num_samples = 20
        d_model = 16
        z_benign = torch.randn(num_samples, d_model)
        
        # Instantiate memory bank
        bank = DeviceMemoryBank(device_class_id=1, k_candidates=[2, 4])
        bank.fit(z_benign)
        
        # Verify optimal K is selected
        self.assertIn(bank.optimal_k, [2, 4])
        self.assertEqual(bank.centroids.shape, (bank.optimal_k, d_model))
        
        # Verify anomaly score output shapes
        z_test = torch.randn(5, d_model)
        scores = bank.compute_anomaly_score(z_test)
        self.assertEqual(scores.shape, (5,))
        # Anomaly scores should be non-negative
        self.assertTrue(torch.all(scores >= 0))

    def test_gate_logic(self):
        gate = AdaptiveInferenceGate()
        
        # Extract features
        H, var_iat = gate.extract_features(self.dummy_input)
        self.assertEqual(H.shape, (self.B,))
        self.assertEqual(var_iat.shape, (self.B,))
        
        # Test TCP flags escalation
        # Create a flow sequence with no flags
        seq_no_flags = torch.zeros(self.B, self.L, self.D_feat)
        escalate_no = gate.check_tcp_escalation(seq_no_flags)
        self.assertFalse(escalate_no.any())
        
        # Set high flag values representing SYN flag escalation
        seq_flags = torch.zeros(self.B, self.L, self.D_feat)
        seq_flags[:, 0, 3] = 0.9 # Flag index 3 set high
        escalate_yes = gate.check_tcp_escalation(seq_flags)
        self.assertTrue(escalate_yes.all())
        
        # Test full gate forward pass
        P_high, escalate = gate(self.dummy_input)
        self.assertEqual(P_high.shape, (self.B,))
        self.assertEqual(escalate.shape, (self.B,))

    def test_quantization_roundtrip_error(self):
        # Test log-scale quantization round-trip error on random parameter weights
        W = torch.randn(10, 10) * 0.5
        # Ensure std is non-zero
        W_quant = log_quantize_dequantize(W, bits=8, outlier_sigma=3.0)
        self.assertEqual(W_quant.shape, W.shape)
        
        # Verify hybrid model quantization
        encoder = MambaEncoder(num_features=self.D_feat, d_model=self.d_model, d_state=8, d_conv=2, num_layers=1)
        quantized = quantize_model(encoder, mode="hybrid")
        # Check that linear layers are replaced with QuantizedLinear
        has_quantized_linear = False
        for m in quantized.modules():
            if "QuantizedLinear" in m.__class__.__name__:
                has_quantized_linear = True
                break
        self.assertTrue(has_quantized_linear)

    def test_runtime_pipeline(self):
        # Build models
        encoder = MambaEncoder(num_features=self.D_feat, d_model=self.d_model, d_state=8, d_conv=2, num_layers=1)
        classifier = ClassifierHead(d_model=self.d_model, n_classes=34)
        fast_classifier = FastClassifier(num_features=self.D_feat, n_classes=34)
        
        # Build memory banks
        memory_banks = {0: DeviceMemoryBank(device_class_id=0)}
        z_dummy = torch.randn(10, self.d_model)
        memory_banks[0].fit(z_dummy)
        
        # Threshold manager
        threshold_manager = DeviceThresholdManager()
        threshold_manager.update_benign(0, 0.2)
        threshold_manager.generic_tracker.update(np.array([0.1, 0.2, 0.3]))
        threshold_manager.generic_tracker.recompute_threshold()
        
        # Normalizer
        normalizer = ZScoreNormalizer(num_features=self.D_feat)
        normalizer.fit(self.dummy_input)
        
        # Instantiate pipeline
        pipeline = DBMambaRuntimePipeline(
            mamba_encoder=encoder,
            classifier_head=classifier,
            fast_classifier=fast_classifier,
            device_mem_banks=memory_banks,
            threshold_manager=threshold_manager,
            normalizer=normalizer,
            device="cpu"
        )
        
        # Process flow sample
        res = pipeline.process_flow(self.dummy_input[0], mac="00:1e:c0:b4:a1:02")
        self.assertIn("path", res)
        self.assertIn("is_anomaly", res)
        self.assertIn("predicted_class", res)
        self.assertIn("anomaly_score", res)

if __name__ == '__main__':
    unittest.main()
