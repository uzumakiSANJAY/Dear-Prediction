CREATE TABLE IF NOT EXISTS draw_results (
  id SERIAL PRIMARY KEY,
  draw_no VARCHAR(20),
  draw_date DATE NOT NULL,
  time_slot VARCHAR(10) NOT NULL,
  prizes JSONB NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(draw_date, time_slot)
);

CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  prediction_date DATE NOT NULL,
  time_slot VARCHAR(10) NOT NULL,
  predicted_numbers JSONB NOT NULL,
  confidence FLOAT,
  model_used VARCHAR(50),
  analysis JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_draw_results_date ON draw_results(draw_date);
CREATE INDEX IF NOT EXISTS idx_draw_results_time ON draw_results(time_slot);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
