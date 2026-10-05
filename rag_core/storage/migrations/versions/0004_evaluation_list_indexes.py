"""Index tenant/project scoped evaluation history lists."""
from alembic import op

revision = "0004_evaluation_list_indexes"
down_revision = "0003_trace_lookup_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_evaluation_datasets_scope_created_at", "evaluation_datasets",
                    ["tenant_id", "project_id", "created_at"], if_not_exists=True)
    op.create_index("ix_evaluation_runs_scope_created_at", "evaluation_runs",
                    ["tenant_id", "project_id", "dataset_id", "created_at"], if_not_exists=True)


def downgrade() -> None:
    op.drop_index("ix_evaluation_runs_scope_created_at", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_datasets_scope_created_at", table_name="evaluation_datasets")
