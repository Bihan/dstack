"""add_gateways_created_by_user_id

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-26 12:00:00.000000+00:00

"""

import sqlalchemy as sa
import sqlalchemy_utils
from alembic import op

# revision identifiers, used by Alembic.
revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("gateways", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "created_by_user_id",
                sqlalchemy_utils.types.uuid.UUIDType(binary=False),
                nullable=True,
            )
        )
        batch_op.create_foreign_key(
            batch_op.f("fk_gateways_created_by_user_id_users"),
            "users",
            ["created_by_user_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("gateways", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("fk_gateways_created_by_user_id_users"),
            type_="foreignkey",
        )
        batch_op.drop_column("created_by_user_id")
