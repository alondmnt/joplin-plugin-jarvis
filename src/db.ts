import joplin from 'api';
import { BlockEmbedding } from './embeddings';
const sqlite3 = joplin.require('sqlite3');
const fs = joplin.require('fs-extra');

// connect to the database
export async function connect_to_db(model: any): Promise<any> {
  await migrate_db();  // migrate the database if necessary
  const plugin_dir = await joplin.plugins.dataDir();
  const db_fname = model.id.replace(/[/\\?%*:|"<>]/g, '_');
  const db = await new sqlite3.Database(plugin_dir + `/${db_fname}.sqlite`);

  // check the model version
  let [check, model_idx] = await db_update_check(db, model);
  let choice = -1;
  if (check === 'new') {
    choice = await joplin.views.dialogs.showMessageBox('The note database is based on a different model. Would you like to rebuild it? (Highly recommended)');
  } else if (check === 'update') {
    choice = await joplin.views.dialogs.showMessageBox('The note database is based on an older version of the model. Would you like to rebuild it? (Recommended)');
  } else if (check === 'size_change') {
    choice = await joplin.views.dialogs.showMessageBox('The note database is based on a different max tokens value. Would you like to rebuild it? (Optional)');
  }

  if (choice === 0) {
    // OK (rebuild)
    db.close();
    await fs.remove(plugin_dir + `/${db_fname}.sqlite`);
    return await connect_to_db(model);

  } else if (choice === 1) {
    // Cancel (keep existing)
    model_idx = await insert_model(db, model);
  }
  model.db_idx = model_idx;

  return db;
}

// create the database tables
export async function init_db(db: any, model: any): Promise<void> {
  if (await db_tables_exist(db)) {
    return;
  }
  // create the table for embeddings
  db.exec(`CREATE TABLE embeddings (
    idx INTEGER PRIMARY KEY,
    line INTEGER NOT NULL,
    body_idx INTEGER NOT NULL,
    length INTEGER NOT NULL,
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
    max_block_size INT NOT NULL,
    embedding_version INTEGER NOT NULL DEFAULT 1,
    UNIQUE (model_name, model_version, max_block_size, embedding_version)
  )`);

  // add the model metadata
  insert_model(db, model);
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

// compare the model metadata in the database to the model metadata in the plugin
async function db_update_check(db: any, model: any): Promise<[String, number]> {
  if (!(await db_tables_exist(db))) {
    return ['OK', 0];
  }
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.all(`SELECT idx, model_name, model_version, max_block_size FROM models`, (err, rows: {idx: number, model_name: string, model_version: string, max_block_size: number}[]) => {
        if (err) {
          reject(err);
        } else {
          // check if the model metadata exists in the table
          // if any of the rows matches the model metadata, then return OK
          let model_exists = false;
          let model_update = true;
          let model_size_change = true;
          let model_idx = 0;

          for (let row of rows) {

            if (row.model_name === model.id) {
              model_exists = true;

              if (row.model_version === model.version) {
                model_update = false;

                if (row.max_block_size === model.max_block_size) {
                  model_size_change = false;
                  model_idx = row.idx;
                }
              }
            }
          }

          if (!model_exists) {
            resolve(['new', 0]);
          }
          if (model_update) {
            resolve(['update', 0]);
          }
          if (model_size_change) {
            resolve(['size_change', 0]);
          } else {
            resolve(['OK', model_idx]);
          }
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
      db.all(`SELECT note_id, hash, line, body_idx, length, level, title, embedding FROM notes JOIN embeddings ON notes.idx = embeddings.note_idx`,
          (err, rows: {note_id: string, hash: string, line: string, body_idx: number, length: number, level: number, title: string, embedding: Buffer}[]) => {
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
              body_idx: row.body_idx,
              length: row.length,
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

function insert_model(db: any, model: any): Promise<number> {
  return new Promise((resolve, reject) => {
    db.run(`INSERT INTO models (model_name, model_version, max_block_size) VALUES ('${model.id}', '${model.version}', ${model.max_block_size})`, function(error) {
      if (error) {
        console.error('connect_to_db error:', error);
        reject(error);
      } else {
        resolve(this.lastID);
      }
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
export async function insert_note_embeddings(db: any, embeds: BlockEmbedding[], model: any): Promise<void> {
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
          const stmt = db.prepare(`INSERT INTO embeddings (note_idx, line, body_idx, length, level, title, embedding, model_idx) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`);
          for (let embd of embeddings) {
            stmt.run([new_row_id, embd.line, embd.body_idx, embd.length, embd.level, embd.title, Buffer.from(embd.embedding.buffer), model.db_idx]);
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

// delete a note and its embeddings from the database.
export async function delete_note_and_embeddings(db: any, note_id: string): Promise<void> {
  const note_status = await get_note_status(db, note_id, '');
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      db.run(`DELETE FROM notes WHERE note_id = ?`, [note_id], (err) => {
        if (err) {
          reject(err);
        } else if (note_status.rowID !== null) {
          db.run(`DELETE FROM embeddings WHERE note_idx = ?`, [note_status.rowID], (err) => {
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

export async function clear_deleted_notes(embeddings: BlockEmbedding[], db: any):
    Promise<BlockEmbedding[]> {
  // get all existing note ids
  let page = 0;
  let notes: any;
  let note_ids = [];
  do {
    page += 1;
    notes = await joplin.data.get(['notes'], { fields: ['id'], page: page });
    note_ids = note_ids.concat(notes.items.map((note: any) => note.id));
  } while(notes.has_more);

  let deleted = [];
  let new_embeddings: BlockEmbedding[] = [];
  for (const embd of embeddings) {

    if (note_ids.includes(embd.id)) {
      new_embeddings.push(embd);

    } else if (!deleted.includes(embd.id)){
      delete_note_and_embeddings(db, embd.id);
      deleted.push(embd.id);
    }
  }

  console.log(`clear_deleted_notes: ${deleted.length} notes removed from DB`);
  return new_embeddings;
}

// migrate the database to the latest version.
async function migrate_db(): Promise<void> {
  const plugin_dir = await joplin.plugins.dataDir();
  const db_path_old = plugin_dir + '/embeddings.sqlite';
  if (!fs.existsSync(db_path_old)) { return; }

  const db_path_new = plugin_dir + '/Universal Sentence Encoder.sqlite';

  console.log(`migrate_db: found old database at ${db_path_old}`);
  fs.renameSync(db_path_old, db_path_new);

  const db = await new sqlite3.Database(db_path_new);
  // alter the models table
  await db.run(`ALTER TABLE models ADD COLUMN max_block_size INT NOT NULL DEFAULT 512`);
  await db.close();
  await new Promise(res => setTimeout(res, 1000));  // make sure that the database is closed
}
