#ifndef HASHMAP_H
#define HASHMAP_H

// Hash map structure for parameters and hyperparameters
typedef struct HashMapEntry {
    char* key;
    void* value;
    struct HashMapEntry* next;
} HashMapEntry;

typedef struct {
    HashMapEntry** buckets;
    int size;
} HashMap;

// HashMap operations
HashMap* hashmap_create(int size);
void hashmap_put(HashMap* map, const char* key, void* value);
void* hashmap_get(HashMap* map, const char* key);
void hashmap_remove(HashMap* map, const char* key);
void hashmap_free(HashMap* map, void (*free_value)(void*));

#endif