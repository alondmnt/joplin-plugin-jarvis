import joplin from 'api';
import { BlockEmbedding } from './embeddings';
const sqlite3 = joplin.require('sqlite3');

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
    id INTEGER PRIMARY KEY,
    line INTEGER NOT NULL,
    level INTEGER NOT NULL,
    title TEXT,
    embedding BLOB NOT NULL
  )`);

  // create the table for note hashes
  db.exec(`CREATE TABLE notes (
    idx INTEGER PRIMARY KEY,
    note_id TEXT NOT NULL,
    hash TEXT NOT NULL
  )`);

  // add a foreign key constraint to connect the two tables
  db.exec(`ALTER TABLE embeddings ADD COLUMN note_idx INTEGER REFERENCES notes(idx)`);
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
  console.log('getting all embeddings...')
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
export async function insert_note_embeddings(db: any, embeds: Promise<BlockEmbedding[]>): Promise<void> {
  const embeddings = await embeds;
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
      if (await note_is_up_to_date(db, embeddings[0].id, embeddings[0].hash)) {
        // no need to update the embeddings
        resolve();
      }
      console.log(`updating embeddings for note ${embeddings[0].id}`)
      const db_id = await insert_note(db, embeddings[0].id, embeddings[0].hash);  // insert or update
      // delete the old embeddings
      db.run(`DELETE FROM embeddings WHERE note_idx = ?`, [db_id], (err) => {
        if (err) {
          reject(err);
        } else {
          // insert the new embeddings
          const stmt = db.prepare(`INSERT INTO embeddings (note_idx, line, level, title, embedding) VALUES (?, ?, ?, ?, ?)`);
          for (let embd of embeddings) {
            stmt.run([db_id, embd.line, embd.level, embd.title, Buffer.from(embd.embedding.buffer)]);
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
export async function note_is_up_to_date(db: any, note_id: string, hash: string): Promise<boolean> {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.get(`SELECT hash FROM notes WHERE note_id = ?`, [note_id], (err, row: {hash: string}) => {
        if (err) {
          reject(err);
        } else if (row) {
          resolve(row.hash === hash);
        } else {
          resolve(false);
        }
      });
    });
  });
}
