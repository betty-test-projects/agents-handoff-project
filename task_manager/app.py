from flask import Flask, jsonify, request, render_template
import sqlite3
import os

app = Flask(__name__)
DATABASE = 'tasks.db'


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── REST API ──────────────────────────────────────────────────────────────────

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM tasks ORDER BY created_at DESC'
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route('/api/tasks', methods=['POST'])
def create_task():
    data = request.get_json()
    title = (data or {}).get('title', '')
    if not title:
        return jsonify({'error': 'Title is required'}), 400
    with get_db() as conn:
        cursor = conn.execute(
            'INSERT INTO tasks (title) VALUES (?)', (title,)
        )
        conn.commit()
        row = conn.execute(
            'SELECT * FROM tasks WHERE id = ?', (cursor.lastrowid,)
        ).fetchone()
    return jsonify(dict(row)), 201


@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    data = request.get_json() or {}
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM tasks WHERE id = ?', (task_id,)
        ).fetchone()
        if row is None:
            return jsonify({'error': 'Task not found'}), 404
        title = data.get('title', row['title']).strip() or row['title']
        completed = int(data['completed']) if 'completed' in data else row['completed']
        conn.execute(
            'UPDATE tasks SET title = ?, completed = ? WHERE id = ?',
            (title, completed, task_id)
        )
        conn.commit()
        updated = conn.execute(
            'SELECT * FROM tasks WHERE id = ?', (task_id,)
        ).fetchone()
    return jsonify(dict(updated))


@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    with get_db() as conn:
        row = conn.execute(
            'SELECT id FROM tasks WHERE id = ?', (task_id,)
        ).fetchone()
        if row is None:
            return jsonify({'error': 'Task not found'}), 404
        conn.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
        conn.commit()
    return jsonify({'message': 'Task deleted'})


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
