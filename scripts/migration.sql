-- Enable required extensions
create extension if not exists vector;

-- Create content type enum
create type content_type as enum (
    'page',
    'documentation',
    'blog',
    'news',
    'api_reference',
    'calendar',
    'other'
);

-- Create the main table for storing document chunks
create table if not exists site_pages (
    id bigserial primary key,
    url text not null,
    chunk_number integer not null,
    title text not null,
    summary text not null,
    content text not null,
    content_type content_type not null default 'page',
    
    -- Sitemap specific fields
    change_frequency text,
    priority decimal(3,2),
    source_created_at timestamp with time zone,
    source_updated_at timestamp with time zone,
    
    -- FIXME Vector search (using 1024 to match BAAI/bge-large-en-v1.5) shold be 768 right?
    embedding vector(1024),
    token_count integer,
    
    -- Path analysis
    path_segments text[] generated always as (
        string_to_array(regexp_replace(url, '^https?://[^/]+', ''), '/')
    ) stored,
    
    -- Metadata and timestamps
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Constraints
    constraint valid_url check (length(url) > 0),
    constraint valid_content check (length(content) > 0),
    constraint valid_priority check (priority >= 0.0 and priority <= 1.0),
    unique(url, chunk_number)
);

-- Optimized indexes
create index idx_site_pages_embedding on site_pages using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);
create index idx_site_pages_metadata on site_pages using gin (metadata);
create index idx_site_pages_path_segments on site_pages using gin (path_segments);
create index idx_site_pages_content_type on site_pages (content_type);
create index idx_site_pages_url_chunk on site_pages (url, chunk_number);

-- Updated similarity search function
create or replace function match_site_pages(
    query_embedding vector(768),
    match_threshold float default 0.3,
    match_count int default 5,
    filter_content_type content_type default null,
    filter_metadata jsonb default '{}'::jsonb,
    filter_path text default null
)
returns table (
    id bigint,
    url text,
    chunk_number integer,
    title text,
    summary text,
    content text,
    content_type content_type,
    metadata jsonb,
    similarity float,
    source_updated_at timestamp with time zone
)
language plpgsql
as $$
begin
    return query
    select
        sp.id,
        sp.url,
        sp.chunk_number,
        sp.title,
        sp.summary,
        sp.content,
        sp.content_type,
        sp.metadata,
        1 - (sp.embedding <=> query_embedding) as similarity,
        sp.source_updated_at
    from site_pages sp
    where
        (filter_content_type is null or sp.content_type = filter_content_type)
        and (filter_metadata = '{}'::jsonb or sp.metadata @> filter_metadata)
        and (filter_path is null or sp.url like '%' || filter_path || '%')
        and (1 - (sp.embedding <=> query_embedding)) > match_threshold
    order by sp.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- Add timestamp trigger
create or replace function update_updated_at()
returns trigger as $$
begin
    new.updated_at = timezone('utc'::text, now());
    return new;
end;
$$ language plpgsql;

create trigger set_updated_at
    before update on site_pages
    for each row
    execute function update_updated_at();

-- Enable RLS and create policies
alter table site_pages enable row level security;

-- Policy for service role (our application backend) - MUST BE FIRST
create policy "service_role_policy"
    on site_pages
    as permissive
    for all
    to postgres, service_role
    using (true)
    with check (true);

-- Policy for authenticated users (read-only)
create policy "authenticated_read_policy"
    on site_pages 
    as permissive
    for select
    to authenticated
    using (true);

-- Helper function for path-based queries
create or replace function get_site_structure()
returns table (
    path text,
    depth int,
    count bigint
)
language sql
as $$
    with recursive paths as (
        select 
            path_segments as segments,
            1 as depth,
            path_segments[1:1] as current_path
        from site_pages
        union all
        select
            p.segments,
            p.depth + 1,
            p.segments[1:p.depth + 1]
        from paths p
        where p.depth < array_length(p.segments, 1)
    )
    select
        '/' || array_to_string(current_path, '/') as path,
        depth,
        count(distinct current_path) as count
    from paths
    group by path, depth, current_path
    order by path;
$$;