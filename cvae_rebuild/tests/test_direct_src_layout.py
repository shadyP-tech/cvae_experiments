from pathlib import Path


def test_no_nested_cvae_rebuild_package_remains() -> None:
    assert not Path("src/cvae_rebuild").exists()


def test_direct_src_imports_resolve() -> None:
    import cli
    import cli_registry
    import config
    import experiments.prior_sampling.posthoc_gmm_pca128 as posthoc_gmm_pca128
    import latent.posterior_latents as posterior_latents
    import priors.gmm as gmm
    import protocol

    assert callable(cli.main)
    assert "diagnose-preservation" in cli_registry.COMMANDS_BY_NAME
    assert hasattr(config, "load_config")
    assert posthoc_gmm_pca128.PCA128_POSTHOC_GMM_NAME == "pca128_posthoc_gmm_prior_v1"
    assert hasattr(posterior_latents, "split_fit_eval_latents")
    assert hasattr(gmm, "fit_class_conditional_gmm_prior")
    assert hasattr(protocol, "ProtocolError")
