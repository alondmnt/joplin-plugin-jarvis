import joplin from 'api';
import { BlockEmbedding } from './embeddings';
const sqlite3 = joplin.require('sqlite3');

// TODO: get from settings or from package
const model = {name: 'Universal Sentence Encoder', version: '1.3.3', id: 1};

// connect to the database
export async function connect_to_db(): Promise<any> {
  const plugin_dir = await joplin.plugins.dataDir();
  return new sqlite3.Database(plugin_dir + '/embeddings.sqlite');
}

// create the database tables
export async function init_db(db: any): Promise<void> {
  if (await db_tables_exist(db)) {
    return;
  }
  // create the table for embeddings
  db.exec(`CREATE TABLE embeddings (
    idx INTEGER PRIMARY KEY,
    line INTEGER NOT NULL,
    level INTEGER NOT NULL,
    title TEXT,
    embedding BLOB NOT NULL,
    note_idx INTEGER NOT NULL REFERENCES notes(idx),
    model_idx INTEGER NOT NULL REFERENCES models(idx)
  )`);

  // create the table for note hashes
  db.exec(`CREATE TABLE notes (
    idx INTEGER PRIMARY KEY,
    note_id TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    UNIQUE (note_id, hash)
  )`);

  // create the table for model metadata
  db.exec(`CREATE TABLE models (
    idx INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    UNIQUE (model_name, model_version)
  )`);

  // add the model metadata
  db.exec(`INSERT INTO models (model_name, model_version) VALUES ('${model.name}', '${model.version}')`);
}

// check if the embeddings and notes tables exist
async function db_tables_exist(db: any): Promise<boolean> {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.all(`SELECT name FROM sqlite_master WHERE type='table'`, (err, rows: {name: string}[]) => {
        if (err) {
          reject(err);
        } else {
          // check if embeddings and notes exist
          let embeddings_exist = false;
          let notes_exist = false;
          for (let row of rows) {
            if (row.name === 'embeddings') {
              embeddings_exist = true;
            }
            if (row.name === 'notes') {
              notes_exist = true;
            }
          }
          resolve(embeddings_exist && notes_exist);
        }
      });
    });
  });
}

// get all the embeddings of all notes from the DB.
// first, join the notes table and the embeddings table.
// then, return all embeddings in a BlockEmbedding array.
export async function get_all_embeddings(db: any): Promise<BlockEmbedding[]> {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.all(`SELECT note_id, hash, line, level, title, embedding FROM notes JOIN embeddings ON notes.idx = embeddings.note_idx`,
          (err, rows: {note_id: string, hash: string, line: string, level: number, title: string, embedding: Buffer}[]) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows.map((row) => {
            // convert the embedding from a blob to a Float32Array
            const buffer = Buffer.from(row.embedding);
            const embedding = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / Float32Array.BYTES_PER_ELEMENT);
            return {
              id: row.note_id,
              hash: row.hash,
              line: parseInt(row.line, 10),
              level: row.level,
              title: row.title,
              embedding: embedding,
              similarity: 0,
            };
          }));
        }
      });
    });
  });
}

// insert a new note into the database, if it already exists update its hash.
// return the id of the note in the database.
export async function insert_note(db: any, note_id: string, hash: string): Promise<number> {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.run(`INSERT OR REPLACE INTO notes (note_id, hash) VALUES (?, ?)`, [note_id, hash], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.lastID);
        }
      });
    });
  });
}

// insert new embeddings for a single note into the database. check if the note hash changed.
// if the hash changed, delete all the embeddings for that note and insert the new ones.
// if the note has no embeddings, insert the new ones.
export async function insert_note_embeddings(db: any, embeds: BlockEmbedding[]): Promise<void> {
  const embeddings = embeds;
  // check that embeddings contain a single note_id
  if (embeddings.length === 0) {
    return;
  }
  for (let embd of embeddings) {
    if ((embd.id !== embeddings[0].id) || (embd.hash !== embeddings[0].hash)) {
      throw new Error('insert_note_embeddings: embeddings contain multiple notes');
    }
  }

  return new Promise((resolve, reject) => {
    db.serialize(async () => {
      const note_status = await get_note_status(db, embeddings[0].id, embeddings[0].hash);
      if (note_status.isUpToDate) {
        // no need to update the embeddings
        resolve();
      }
      const new_row_id = await insert_note(db, embeddings[0].id, embeddings[0].hash);  // insert or update
      // delete the old embeddings
      db.run(`DELETE FROM embeddings WHERE note_idx = ?`, [note_status.rowID], (err) => {
        if (err) {
          reject(err);
        } else {
          // insert the new embeddings
          const stmt = db.prepare(`INSERT INTO embeddings (note_idx, line, level, title, embedding, model_idx) VALUES (?, ?, ?, ?, ?, ?)`);
          for (let embd of embeddings) {
            stmt.run([new_row_id, embd.line, embd.level, embd.title, Buffer.from(embd.embedding.buffer), model.id]);
          }
          stmt.finalize();
          resolve();
        }
      });
    });
  });
}

// check if a note exists in the database.
// if it does, compare its hash with the hash of the note in the database.
// if the hashes are the same return true, otherwise return false.
// if it does not exist in the database, return false.
export async function get_note_status(db: any, note_id: string, hash: string): Promise<{isUpToDate: boolean, rowID: number | null}> {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.get(`SELECT idx, hash FROM notes WHERE note_id = ?`, [note_id], (err, row: {idx: number, hash: string}) => {
        if (err) {
          reject(err);
        } else if (row) {
          resolve({ isUpToDate: row.hash === hash, rowID: row.idx });
        } else {
          resolve({ isUpToDate: false, rowID: null });
        }
      });
    });
  });
}

// delete everything from DB
export async function clear_db(db: any): Promise<void> {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.run(`DELETE FROM embeddings`, (err) => {
        if (err) {
          reject(err);
        } else {
          db.run(`DELETE FROM notes`, (err) => {
            if (err) {
              reject(err);
            } else {
              resolve();
            }
          });
        }
      });
    });
  });
}
