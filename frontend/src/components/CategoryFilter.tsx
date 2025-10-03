import { Button } from '@/components/ui/button';

interface CategoryFilterProps {
  categories: string[];
  selectedCategory: string | null;
  onSelectCategory: (category: string | null) => void;
}

const CategoryFilter = ({
  categories,
  selectedCategory,
  onSelectCategory,
}: CategoryFilterProps) => {
  return (
    <div className="flex flex-wrap gap-2 mb-8 animate-fade-in">
      <Button
        variant={selectedCategory === null ? 'default' : 'outline'}
        size="sm"
        onClick={() => onSelectCategory(null)}
        className={selectedCategory === null ? 'bg-accent hover:bg-accent/90' : ''}
      >
        All
      </Button>
      {categories.map((category) => (
        <Button
          key={category}
          variant={selectedCategory === category ? 'default' : 'outline'}
          size="sm"
          onClick={() => onSelectCategory(category)}
          className={selectedCategory === category ? 'bg-accent hover:bg-accent/90' : ''}
        >
          {category}
        </Button>
      ))}
    </div>
  );
};

export default CategoryFilter;
