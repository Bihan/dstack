"""Add InstanceModel.node_group_index

Revision ID: b1a2c3d4e5f6
Revises: a13f5b55af01
Create Date: 2026-03-09 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b1a2c3d4e5f6"
down_revision = "a13f5b55af01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("instances", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("node_group_index", sa.Integer(), nullable=False, server_default="0")
        )


def downgrade() -> None:
    with op.batch_alter_table("instances", schema=None) as batch_op:
        batch_op.drop_column("node_group_index")
