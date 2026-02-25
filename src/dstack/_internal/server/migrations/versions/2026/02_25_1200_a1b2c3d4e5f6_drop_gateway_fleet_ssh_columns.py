"""drop_gateway_fleet_ssh_columns

Revision ID: a1b2c3d4e5f6
Revises: 418fc659954a
Create Date: 2026-02-25 12:00:00.000000+00:00

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "418fc659954a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("gateways", schema=None) as batch_op:
        batch_op.drop_column("fleet_ssh_public_key")
        batch_op.drop_column("fleet_ssh_private_key")


def downgrade() -> None:
    with op.batch_alter_table("gateways", schema=None) as batch_op:
        batch_op.add_column(sa.Column("fleet_ssh_private_key", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("fleet_ssh_public_key", sa.Text(), nullable=True))
