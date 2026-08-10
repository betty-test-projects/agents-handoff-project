# Recurring Tasks

*Spec v1*

## Background

The current task manager (`app.py` + `templates/index.html`) supports one-off tasks: 
create, complete, edit, delete. Users have asked for tasks that repeat on a 
schedule — e.g. "water the plants every day", "team sync every Monday", 
"pay rent on the 1st of each month". This document describes the recurring 
tasks feature.

## Goals

- Let users mark a task as recurring with a daily, weekly, or monthly cadence.
- The recurring task should re-appear in the active list when it is next due.
- The feature should integrate with the existing UI without disrupting how 
  one-off tasks work today.

## Functional Requirements

### FR-1. Recurrence types

A task can have one of four recurrence settings:

- `none` (default, behaves exactly like today's one-off tasks)
- `daily`
- `weekly`  
- `monthly`

Recurrence is set at task creation and can be edited later.

### FR-2. Data model

The `tasks` table gains the following columns:

- `recurrence` TEXT — one of `none`, `daily`, `weekly`, `monthly`. Default `none`.
- `next_due` TIMESTAMP — when the task is next due to appear as active. 
  NULL for `recurrence = none`.

### FR-3. Creating a recurring task

When a user creates a task and selects a recurrence other than `none`, the 
`next_due` field is set:

- `daily` → tomorrow
- `weekly` → seven days from today
- `monthly` → the same day of next month

### FR-4. Completing a recurring task

When a user marks a recurring task as completed, the task is deleted from 
the database. A new task with the same title and recurrence is then created 
with `next_due` advanced according to the recurrence type.

### FR-5. Editing a recurring task

A user can edit the title of a recurring task at any time. The edit takes 
effect immediately.

### FR-6. Listing behaviour

The active list (`/api/tasks` GET) shows:

- All one-off tasks that are not completed.
- All recurring tasks whose `next_due` is today or earlier.

Recurring tasks whose `next_due` is in the future are not shown in the 
active list.

### FR-7. Filter compatibility

The existing `ALL` / `ACTIVE` / `DONE` filters continue to work. Recurring 
tasks that are currently due appear under `ACTIVE`. The `DONE` filter shows 
completed tasks as before.

## UI Notes

- The "Add task" input gains a small dropdown next to it for selecting 
  recurrence (`none`, `daily`, `weekly`, `monthly`). Default is `none`.
- Recurring tasks in the list show a small indicator (e.g. a ↻ symbol) 
  next to the title.
- The edit flow continues to use the existing inline edit input. No 
  recurrence-specific editing UI is needed in this version.

## Out of Scope

- Custom recurrence patterns (e.g. "every 3 days", "every other Tuesday").
- Notifications or reminders.
- Calendar view.
- Per-user accounts (the app remains single-user as today).

## Files to modify

- `app.py` — schema, API endpoints, recurrence logic.
- `templates/index.html` — UI for selecting and displaying recurrence.
