#include "../include/hashmap.h"
#include <stdlib.h>
#include <string.h>

#define HASHMAP_DEFAULT_SIZE 16

// Hash function for string keys
static unsigned int hash_string(const char* key, int size) {
    unsigned int hash = 5381;
    for (int i = 0; key[i] != '\0'; i++) {
        hash = ((hash << 5) + hash) + key[i];
    }
    return hash % size;
}

// HashMap operations
HashMap* hashmap_create(int size) {
    HashMap* map = (HashMap*)malloc(sizeof(HashMap));
    if (!map) return NULL;
    
    map->size = size;
    map->buckets = (HashMapEntry**)calloc(size, sizeof(HashMapEntry*));
    if (!map->buckets) {
        free(map);
        return NULL;
    }
    
    return map;
}

void hashmap_put(HashMap* map, const char* key, void* value) {
    if (!map || !key) return;
    
    unsigned int index = hash_string(key, map->size);
    HashMapEntry* entry = map->buckets[index];
    
    // Check if key already exists
    while (entry != NULL) {
        if (strcmp(entry->key, key) == 0) {
            entry->value = value;
            return;
        }
        entry = entry->next;
    }
    
    // Create new entry
    HashMapEntry* new_entry = (HashMapEntry*)malloc(sizeof(HashMapEntry));
    if (!new_entry) return;
    
    new_entry->key = (char*)malloc((strlen(key) + 1) * sizeof(char));
    strcpy(new_entry->key, key);
    new_entry->value = value;
    new_entry->next = map->buckets[index];
    map->buckets[index] = new_entry;
}

void* hashmap_get(HashMap* map, const char* key) {
    if (!map || !key) return NULL;
    
    unsigned int index = hash_string(key, map->size);
    HashMapEntry* entry = map->buckets[index];
    
    while (entry != NULL) {
        if (strcmp(entry->key, key) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }
    
    return NULL;
}

void hashmap_remove(HashMap* map, const char* key) {
    if (!map || !key) return;
    
    unsigned int index = hash_string(key, map->size);
    HashMapEntry* entry = map->buckets[index];
    HashMapEntry* prev = NULL;
    
    while (entry != NULL) {
        if (strcmp(entry->key, key) == 0) {
            if (prev == NULL) {
                map->buckets[index] = entry->next;
            } else {
                prev->next = entry->next;
            }
            free(entry->key);
            free(entry);
            return;
        }
        prev = entry;
        entry = entry->next;
    }
}

void hashmap_free(HashMap* map, void (*free_value)(void*)) {
    if (!map) return;
    
    for (int i = 0; i < map->size; i++) {
        HashMapEntry* entry = map->buckets[i];
        while (entry != NULL) {
            HashMapEntry* next = entry->next;
            if (free_value && entry->value) {
                free_value(entry->value);
            }
            free(entry->key);
            free(entry);
            entry = next;
        }
    }
    
    free(map->buckets);
    free(map);
}