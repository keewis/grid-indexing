# Terminology

```{glossary}
arrow-array
    A arrow array, as defined by the [Apache Arrow] specification, and as
    implemented by e.g. [arro3].

    [Apache Arrow]: https://arrow.apache.org
    [arro3]: https://github.com/kylebarron/arro3

sparse-array
    A sparse array is mostly empty or filled with a background value (e.g. 0),
    with only a few values that are different. To reduce the memory footprint,
    sparse array objects usually only store the background value once and instead
    record the location of each foreground value, in various different formats.

    See also [pydata sparse](https://sparse.pydata.org) and {doc}`scipy:tutorial/sparse`

chunked-sparse-array
    A {term}`sparse-array` that was wrapped by a chunked array implementation
    (e.g., a {py:class}`dask array <dask.array.Array>`).
```
