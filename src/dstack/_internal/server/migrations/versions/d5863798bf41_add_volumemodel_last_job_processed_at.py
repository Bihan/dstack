"""Add VolumeModel.last_job_processed_at

Revision ID: d5863798bf41
Revises: 644b8a114187
Create Date: 2025-07-15 14:26:22.981687

"""

import sqlalchemy as sa
from alembic import op

import dstack._internal.server.models

# revision identifiers, used by Alembic.
revision = "d5863798bf41"
down_revision = "644b8a114187"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("volumes", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "last_job_processed_at",
                dstack._internal.server.models.NaiveDateTime(),
                nullable=True,
            )
        )

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("volumes", schema=None) as batch_op:
        batch_op.drop_column("last_job_processed_at")

    # ### end Alembic commands ###
