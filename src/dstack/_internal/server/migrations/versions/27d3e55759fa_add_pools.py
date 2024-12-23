"""add pools

Revision ID: 27d3e55759fa
Revises: d3e8af4786fa
Create Date: 2024-02-12 14:27:52.035476

"""

import sqlalchemy as sa
import sqlalchemy_utils
from alembic import op

# revision identifiers, used by Alembic.
revision = "27d3e55759fa"
down_revision = "d3e8af4786fa"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "pools",
        sa.Column("id", sqlalchemy_utils.types.uuid.UUIDType(binary=False), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.Column(
            "project_id", sqlalchemy_utils.types.uuid.UUIDType(binary=False), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_pools_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_pools")),
    )
    op.create_table(
        "instances",
        sa.Column("id", sqlalchemy_utils.types.uuid.UUIDType(binary=False), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("deleted", sa.Boolean(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.Column(
            "project_id", sqlalchemy_utils.types.uuid.UUIDType(binary=False), nullable=False
        ),
        sa.Column("pool_id", sqlalchemy_utils.types.uuid.UUIDType(binary=False), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "CREATING",
                "STARTING",
                "READY",
                "BUSY",
                "TERMINATING",
                "TERMINATED",
                "FAILED",
                name="instancestatus",
            ),
            nullable=False,
        ),
        sa.Column("status_message", sa.String(length=50), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("termination_policy", sa.String(length=50), nullable=True),
        sa.Column("termination_idle_time", sa.Integer(), nullable=False),
        sa.Column(
            "backend",
            sa.Enum(
                "AWS",
                "AZURE",
                "DATACRUNCH",
                "DSTACK",
                "GCP",
                "KUBERNETES",
                "LAMBDA",
                "LOCAL",
                "REMOTE",
                "NEBIUS",
                "TENSORDOCK",
                "VASTAI",
                "VULTR",
                name="backendtype",
            ),
            nullable=False,
        ),
        sa.Column("backend_data", sa.String(length=4000), nullable=True),
        sa.Column("region", sa.String(length=2000), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("job_provisioning_data", sa.String(length=4000), nullable=False),
        sa.Column("offer", sa.String(length=4000), nullable=False),
        sa.Column("resource_spec_data", sa.String(length=4000), nullable=True),
        sa.Column("job_id", sqlalchemy_utils.types.uuid.UUIDType(binary=False), nullable=True),
        sa.Column("last_job_processed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name=op.f("fk_instances_job_id_jobs")),
        sa.ForeignKeyConstraint(
            ["pool_id"], ["pools.id"], name=op.f("fk_instances_pool_id_pools")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_instances_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_instances")),
    )
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "used_instance_id",
                sqlalchemy_utils.types.uuid.UUIDType(binary=False),
                nullable=True,
            )
        )

    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "default_pool_id",
                sqlalchemy_utils.types.uuid.UUIDType(binary=False),
                nullable=True,
            )
        )
        batch_op.create_foreign_key(
            batch_op.f("fk_projects_default_pool_id_pools"),
            "pools",
            ["default_pool_id"],
            ["id"],
            ondelete="SET NULL",
            use_alter=True,
        )

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("fk_projects_default_pool_id_pools"), type_="foreignkey"
        )
        batch_op.drop_column("default_pool_id")

    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("used_instance_id")

    op.drop_table("instances")
    op.drop_table("pools")
    # ### end Alembic commands ###
