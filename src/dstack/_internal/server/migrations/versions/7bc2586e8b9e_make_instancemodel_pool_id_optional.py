"""Make InstanceModel.pool_id optional

Revision ID: 7bc2586e8b9e
Revises: bc8ca4a505c6
Create Date: 2025-03-13 11:13:39.748303

"""

import sqlalchemy_utils
from alembic import op

# revision identifiers, used by Alembic.
revision = "7bc2586e8b9e"
down_revision = "bc8ca4a505c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("instances", schema=None) as batch_op:
        batch_op.alter_column(
            "pool_id", existing_type=sqlalchemy_utils.UUIDType(binary=False), nullable=True
        )

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("instances", schema=None) as batch_op:
        batch_op.alter_column(
            "pool_id", existing_type=sqlalchemy_utils.UUIDType(binary=False), nullable=False
        )

    # ### end Alembic commands ###
