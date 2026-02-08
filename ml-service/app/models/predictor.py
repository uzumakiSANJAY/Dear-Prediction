import logging
from datetime import date
from .trainer import EnsembleTrainer

logger = logging.getLogger(__name__)

# Global ensemble instance per time slot
_trainers = {}


def get_trainer(time_slot: str) -> EnsembleTrainer:
    if time_slot not in _trainers:
        _trainers[time_slot] = EnsembleTrainer()
    return _trainers[time_slot]


def train_model(time_slot: str, draw_data: list) -> dict:
    """Train the ensemble model for a specific time slot."""
    trainer = get_trainer(time_slot)
    success = trainer.train(draw_data)
    return {
        "status": "trained" if success else "failed",
        "time_slot": time_slot,
        "records_used": len(draw_data),
    }


def predict(time_slot: str) -> dict:
    """Generate predictions for a time slot. Same day = same result."""
    trainer = get_trainer(time_slot)

    if not trainer.is_trained:
        logger.warning(f"Model not trained for {time_slot}, returning random predictions")

    today = str(date.today())
    result = trainer.predict(time_slot=time_slot, date_str=today)
    result["time_slot"] = time_slot
    result["prediction_date"] = today

    return result
