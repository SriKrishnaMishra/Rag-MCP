"""Index tenant trace history for recent-first inspection."""
from alembic import op

revision = "0003_trace_lookup_index"
down_revision = "0002_evaluation_datasets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The initial schema uses current metadata for listed tables, so fresh installs
    # may already have this index before this incremental migration is applied.
    op.create_index("ix_traces_tenant_created_at", "traces", ["tenant_id", "created_at"], if_not_exists=True)


def downgrade() -> None:
    op.drop_index("ix_traces_tenant_created_at", table_name="traces")
