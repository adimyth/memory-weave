# Configuration

One YAML file, loaded by `load_config`; every value has a default, and validation rejects an inconsistent file. The reasoning behind each value is in [section 2 of the low-level design](../agent-memory-lld.md#2-configuration).

::: retold.config
    options:
      members:
        - load_config
        - RetoldConfig
        - StoreConfig
        - EmbeddingConfig
        - RerankerConfig
        - RetrievalConfig
        - RewriteConfig
        - TriggerConfig
        - GateConfig
        - AutoGateConfig
        - DenseFloorConfig
        - FreshnessConfig
        - IngestionConfig
        - EquivalenceConfig
        - EvidenceConfig
        - PolicyConfig
        - SourceRankConfig
        - SessionsConfig
        - ConfigError
