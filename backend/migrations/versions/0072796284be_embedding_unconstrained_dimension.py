"""embedding_unconstrained_dimension

Revision ID: 0072796284be
Revises: ef07882c3816
Create Date: 2026-10-20 15:56:12.238096

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0072796284be'
down_revision: Union[str, Sequence[str], None] = 'ef07882c3816'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: The embedding column was constrained to 2048 dimensions (pgvector's index
#: limit), but tests and different embedding models use other dimensions.
#: Removing the constraint lets any vector width be stored. Vector search
#: uses linear scan for personal use, so no index is needed yet.
EMBEDDING_COLUMN = "embedding"
CHUNKS_TABLE = "chunks"


def upgrade() -> None:
    """Remove the dimension constraint on the embedding column."""
    op.execute(
        f"ALTER TABLE {CHUNKS_TABLE} ALTER COLUMN {EMBEDDING_COLUMN} TYPE vector"
    )


def downgrade() -> None:
    """Re-apply the 2048-dimension constraint (may fail if data exceeds it)."""
    op.execute(
        f"ALTER TABLE {CHUNKS_TABLE} ALTER COLUMN {EMBEDDING_COLUMN} TYPE vector(2048)"
    )
