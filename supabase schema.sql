-- supabase_schema.sql
-- Run in Supabase → SQL Editor → New query → Run
-- Safe to re-run (IF NOT EXISTS throughout)

-- ── Alerts ───────────────────────────────────────────────────────────────────
create table if not exists alerts (
  id                bigserial primary key,
  ts                timestamptz not null default now(),
  session_id        text        not null,
  device_id         text,
  message           text        not null,
  priority          int         not null,
  zone_left         text,
  zone_center       text,
  zone_right        text,
  closest_class     text,
  closest_region    text,
  closest_proximity text
);
create index if not exists alerts_session_idx on alerts(session_id, ts desc);
create index if not exists alerts_device_idx  on alerts(device_id,  ts desc);

-- ── Sensor readings ───────────────────────────────────────────────────────────
create table if not exists sensor_readings (
  id              bigserial primary key,
  ts              timestamptz not null default now(),
  session_id      text        not null,
  device_id       text,
  sensor_cm       float,
  sensor_band     text,
  object_count    int,
  confirmed_count int
);
create index if not exists sensor_session_idx on sensor_readings(session_id, ts desc);

-- ── Commands ──────────────────────────────────────────────────────────────────
-- Cloud dashboard writes rows here; Pi polls and executes them.
-- Supported command values: STATUS | SET_VOLUME | NIGHT_MODE | REQUEST_IMAGE
create table if not exists commands (
  id          bigserial   primary key,
  created_at  timestamptz not null default now(),
  executed_at timestamptz,
  device_id   text        not null,
  command     text        not null,
  payload     jsonb,
  executed    boolean     not null default false
);
create index if not exists commands_device_idx
  on commands(device_id, executed, created_at asc);

-- ── Row Level Security ────────────────────────────────────────────────────────
alter table alerts          enable row level security;
alter table sensor_readings enable row level security;
alter table commands        enable row level security;

-- Read (dashboard)
create policy if not exists "anon read alerts"
  on alerts for select using (true);
create policy if not exists "anon read sensor"
  on sensor_readings for select using (true);
create policy if not exists "anon read commands"
  on commands for select using (true);

-- Insert (Pi writes with anon key)
create policy if not exists "anon insert alerts"
  on alerts for insert with check (true);
create policy if not exists "anon insert sensor"
  on sensor_readings for insert with check (true);
create policy if not exists "anon insert commands"
  on commands for insert with check (true);

-- Update (Pi marks commands executed; dashboard marks executed)
create policy if not exists "anon update commands"
  on commands for update using (true);